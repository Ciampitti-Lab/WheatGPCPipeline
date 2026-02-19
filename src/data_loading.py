"""Data loading functions for spectral, meteorological, soil, and elevation data."""

import glob
import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import pandas as pd

logger = logging.getLogger("gpc_pipeline")


def load_spectral(spectral_dir: str) -> pd.DataFrame:
    """Load and concatenate all monthly Sentinel-2 CSV files."""
    files = sorted(glob.glob(f"{spectral_dir}/wheat_daily_s2_*.csv"))
    if not files:
        raise FileNotFoundError(f"No spectral CSVs found in {spectral_dir}")

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["field_key", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    n_fields = df["field_key"].nunique()
    n_dates = df["date"].nunique()
    logger.info(
        "Spectral data loaded: %d fields, %d unique dates, %d rows, %d files",
        n_fields, n_dates, len(df), len(files),
    )
    return df


def load_meteo(meteo_dir: str) -> pd.DataFrame:
    """Load and concatenate all monthly meteo CSV files."""
    files = sorted(glob.glob(f"{meteo_dir}/wheat_daily_meteo_*.csv"))
    if not files:
        raise FileNotFoundError(f"No meteo CSVs found in {meteo_dir}")

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["field_key", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(
        "Meteo data loaded: %d fields, %d unique dates, %d rows, %d files",
        df["field_key"].nunique(), df["date"].nunique(), len(df), len(files),
    )
    return df


def _fill_missing_counties(
    df: pd.DataFrame,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Fill missing county/state via spatial join with Census Bureau shapefile."""
    required = {"centroid_lat", "centroid_lon", "county", "state"}
    if not required.issubset(df.columns):
        return df

    mask = (
        df["county"].isna()
        & df["centroid_lat"].notna()
        & df["centroid_lon"].notna()
    )
    n_missing = mask.sum()
    if n_missing == 0:
        return df

    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        logger.warning(
            "geopandas not installed -- cannot fill %d missing county values. "
            "Install with: pip install geopandas",
            n_missing,
        )
        return df

    logger.info("Filling %d missing county/state values via spatial join", n_missing)

    # Resolve cache directory
    if cache_dir is None:
        from .utils import get_project_root
        cache_dir = str(get_project_root() / "data" / "cache")
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Download shapefile if not cached
    shp_dir = cache_path / "cb_2022_us_county_500k"
    shp_file = shp_dir / "cb_2022_us_county_500k.shp"

    if not shp_file.exists():
        url = (
            "https://www2.census.gov/geo/tiger/GENZ2022/shp/"
            "cb_2022_us_county_500k.zip"
        )
        zip_path = cache_path / "cb_2022_us_county_500k.zip"
        logger.info("Downloading US county shapefile from Census Bureau...")
        try:
            urlretrieve(url, zip_path)
        except Exception as e:
            logger.warning(
                "Failed to download county shapefile: %s. "
                "County/state values will remain NaN.",
                e,
            )
            return df
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(shp_dir)
        zip_path.unlink()
        logger.info("County shapefile cached at %s", shp_dir)

    # Load counties and build point geometries for missing rows
    counties_gdf = gpd.read_file(shp_file)
    missing_df = df.loc[mask]
    geometry = [
        Point(lon, lat)
        for lon, lat in zip(missing_df["centroid_lon"], missing_df["centroid_lat"])
    ]
    points_gdf = gpd.GeoDataFrame(
        missing_df[["centroid_lat", "centroid_lon"]],
        geometry=geometry,
        crs="EPSG:4326",
    )
    points_gdf = points_gdf.to_crs(counties_gdf.crs)

    # Spatial join
    joined = gpd.sjoin(points_gdf, counties_gdf, how="left", predicate="within")

    # Fill county and state back into original DataFrame
    for idx in joined.index:
        county_name = joined.loc[idx, "NAME"]
        state_name = joined.loc[idx, "STATE_NAME"]
        if pd.notna(county_name):
            df.loc[idx, "county"] = county_name
        if pd.notna(state_name):
            df.loc[idx, "state"] = state_name

    n_filled = n_missing - df.loc[mask, "county"].isna().sum()
    logger.info("Filled %d/%d missing county/state values", n_filled, n_missing)

    return df


def load_static_features(
    features_path: str,
    soil_columns: Optional[List[str]] = None,
    static_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load static features and protein target. Returns (static_df, protein_df)."""
    df = pd.read_csv(features_path)

    # Fill missing county/state via reverse geocoding
    df = _fill_missing_counties(df)

    # Gather soil columns
    soil_cols = []
    if soil_columns:
        soil_cols = [c for c in soil_columns if c in df.columns]
        missing = set(soil_columns) - set(soil_cols)
        if missing:
            logger.warning("Soil columns not found: %s", missing)

    # Gather extra static columns
    extra_cols = []
    if static_columns:
        extra_cols = [c for c in static_columns if c in df.columns]
        missing = set(static_columns) - set(extra_cols)
        if missing:
            logger.warning("Static columns not found: %s", missing)

    # Always include location columns if available
    location_cols = ["centroid_lat", "centroid_lon", "county", "state"]
    location_cols = [c for c in location_cols if c in df.columns]

    all_feature_cols = list(dict.fromkeys(soil_cols + extra_cols + location_cols))
    static_df = df[["field_key"] + all_feature_cols].copy()

    # Protein target
    protein_df = df[["field_key", "protein_pct"]].copy()
    protein_df.dropna(subset=["protein_pct"], inplace=True)

    # Include yield if available
    if "yield_bu_ac" in df.columns:
        static_df["yield_bu_ac"] = df["yield_bu_ac"]

    logger.info(
        "Static features: %d fields, %d columns (soil=%d, static=%d, location=%d)",
        len(static_df), len(all_feature_cols),
        len(soil_cols), len(extra_cols), len(location_cols),
    )
    logger.info("Protein target: %d fields with valid protein_pct", len(protein_df))

    return static_df, protein_df


def load_elevation(
    field_elev_path: str,
    sample_elev_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load field-level elevation, slope, aspect stats."""
    df = pd.read_csv(field_elev_path)

    # Standardize column names
    rename = {}
    if "centroid_l" in df.columns:
        rename["centroid_l"] = "centroid_lon"
    if "centroid_1" in df.columns:
        rename["centroid_1"] = "centroid_lat"
    if "elev_mean" in df.columns:
        rename["elev_mean"] = "elevation"
    if "slope_mean" in df.columns:
        rename["slope_mean"] = "slope"
    if "aspect_mean" in df.columns:
        rename["aspect_mean"] = "aspect"
    df.rename(columns=rename, inplace=True)

    # Select relevant columns
    elev_stats = ["elevation", "slope", "aspect"]
    extra_stats = [
        "elev_std", "elev_range", "elev_min", "elev_max",
        "elev_p10", "elev_p25", "elev_p75", "elev_p90",
        "slope_std", "slope_min", "slope_max", "aspect_std",
    ]
    keep = ["field_key"] + [c for c in elev_stats + extra_stats if c in df.columns]
    result = df[keep].copy()

    # Merge sample-level elevation if provided
    if sample_elev_path:
        sample_df = pd.read_csv(sample_elev_path)
        sample_rename = {}
        if "elevation" in sample_df.columns and "elevation" in result.columns:
            sample_rename["elevation"] = "sample_elevation"
        if "slope" in sample_df.columns and "slope" in result.columns:
            sample_rename["slope"] = "sample_slope"
        if "aspect" in sample_df.columns and "aspect" in result.columns:
            sample_rename["aspect"] = "sample_aspect"
        sample_df.rename(columns=sample_rename, inplace=True)

        sample_cols = ["field_key"] + [
            c for c in sample_df.columns
            if c.startswith("sample_") and c in sample_df.columns
        ]
        if len(sample_cols) > 1:
            result = result.merge(
                sample_df[sample_cols], on="field_key", how="left",
            )

    logger.info("Elevation data: %d fields, %d columns", len(result), len(result.columns) - 1)
    return result


def merge_all_static(
    static_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    elevation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge soil, protein, and elevation into a single DataFrame."""
    merged = static_df.merge(protein_df, on="field_key", how="inner")

    # Avoid duplicate columns when merging elevation
    elev_cols = [c for c in elevation_df.columns if c != "field_key"]
    existing = [c for c in elev_cols if c in merged.columns]
    if existing:
        elevation_df = elevation_df.drop(columns=existing, errors="ignore")

    merged = merged.merge(elevation_df, on="field_key", how="left")

    logger.info(
        "Merged static features: %d fields, %d columns",
        len(merged), len(merged.columns),
    )
    return merged
