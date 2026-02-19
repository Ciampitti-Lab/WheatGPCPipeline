"""Feature engineering functions for the GPC Pipeline."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("gpc_pipeline")

EPSILON = 1e-6


# ---------------------------------------------------------------------------
# Temporal derivatives
# ---------------------------------------------------------------------------

def create_temporal_derivatives(
    df: pd.DataFrame,
    periods: List[str],
    feature_names: List[str],
) -> pd.DataFrame:
    """Create change, ratio, and rate-of-change features between consecutive periods."""
    out = df.copy()
    n_created = 0

    for i in range(len(periods) - 1):
        p1, p2 = periods[i], periods[i + 1]

        for feat in feature_names:
            # Find matching columns for this feature in both periods
            col1_mean = f"{p1}_{feat}_mean"
            col2_mean = f"{p2}_{feat}_mean"

            if col1_mean not in out.columns or col2_mean not in out.columns:
                continue

            v1 = out[col1_mean]
            v2 = out[col2_mean]

            # Change
            out[f"{feat}_change_{p1}_{p2}"] = v2 - v1
            n_created += 1

            # Ratio
            ratio = v2 / (v1.abs() + EPSILON)
            out[f"{feat}_ratio_{p1}_{p2}"] = ratio.clip(-10, 10)
            n_created += 1

            # Rate of change
            roc = (v2 - v1) / (v1.abs() + EPSILON)
            out[f"{feat}_roc_{p1}_{p2}"] = roc.clip(-10, 10)
            n_created += 1

    logger.info("Temporal derivatives: %d features created", n_created)
    return out


# ---------------------------------------------------------------------------
# Band ratios
# ---------------------------------------------------------------------------

def create_band_ratios(
    df: pd.DataFrame,
    periods: List[str],
    ratio_pairs: List[List[str]],
) -> pd.DataFrame:
    """Create spectral band ratio features (b1/b2) per period."""
    out = df.copy()
    n_created = 0

    for period in periods:
        for b1, b2 in ratio_pairs:
            col1 = f"{period}_{b1}_mean"
            col2 = f"{period}_{b2}_mean"

            if col1 not in out.columns or col2 not in out.columns:
                continue

            ratio_name = f"{period}_{b1}_div_{b2}"
            out[ratio_name] = out[col1] / (out[col2].abs() + EPSILON)
            out[ratio_name] = out[ratio_name].clip(-100, 100)
            n_created += 1

    logger.info("Band ratios: %d features created", n_created)
    return out


# ---------------------------------------------------------------------------
# VI interactions
# ---------------------------------------------------------------------------

def create_vi_interactions(
    df: pd.DataFrame,
    periods: List[str],
    interaction_pairs: List[List[str]],
) -> pd.DataFrame:
    """Create VI x VI interaction features (products) per period."""
    out = df.copy()
    n_created = 0

    for period in periods:
        for vi1, vi2 in interaction_pairs:
            col1 = f"{period}_{vi1}_mean"
            col2 = f"{period}_{vi2}_mean"

            if col1 not in out.columns or col2 not in out.columns:
                continue

            out[f"{period}_{vi1}_x_{vi2}"] = out[col1] * out[col2]
            n_created += 1

    logger.info("VI interactions: %d features created", n_created)
    return out


# ---------------------------------------------------------------------------
# Agronomic indices
# ---------------------------------------------------------------------------

def create_agronomic_indices(
    df: pd.DataFrame,
    periods: List[str],
) -> pd.DataFrame:
    """Create aridity index, water deficit, and heat stress per period."""
    out = df.copy()
    n_created = 0

    for period in periods:
        precip_col = f"{period}_precip_sum"
        pet_col = f"{period}_pet_sum"
        tmax_col = f"{period}_t2m_max_mean"

        # Aridity index
        if precip_col in out.columns and pet_col in out.columns:
            out[f"{period}_aridity_idx"] = out[precip_col] / (out[pet_col].abs() + EPSILON)
            out[f"{period}_water_deficit"] = out[pet_col] - out[precip_col]
            n_created += 2

        # Heat stress
        if tmax_col in out.columns:
            out[f"{period}_heat_stress"] = (out[tmax_col] - 30).clip(lower=0)
            n_created += 1

    logger.info("Agronomic indices: %d features created", n_created)
    return out


# ---------------------------------------------------------------------------
# VI x Meteo interactions
# ---------------------------------------------------------------------------

def create_vi_meteo_interactions(
    df: pd.DataFrame,
    periods: List[str],
    vi_list: List[str],
    meteo_vars: List[str],
    critical_periods: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create VI x Meteo interaction features at critical periods."""
    out = df.copy()
    n_created = 0

    target_periods = critical_periods if critical_periods else periods

    for period in target_periods:
        if period not in periods:
            continue
        for vi in vi_list:
            vi_col = f"{period}_{vi}_mean"
            if vi_col not in out.columns:
                continue
            for meteo in meteo_vars:
                # Try different suffixes
                for suffix in ["_mean", "_sum", ""]:
                    meteo_col = f"{period}_{meteo}{suffix}"
                    if meteo_col in out.columns:
                        out[f"{period}_{vi}_x_{meteo}"] = out[vi_col] * out[meteo_col]
                        n_created += 1
                        break

    logger.info("VI x Meteo interactions: %d features created", n_created)
    return out


# ---------------------------------------------------------------------------
# Bands x VI interactions
# ---------------------------------------------------------------------------

def create_band_vi_interactions(
    df: pd.DataFrame,
    periods: List[str],
    band_list: List[str],
    vi_list: List[str],
    critical_periods: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create Band x VI interaction features at critical periods."""
    out = df.copy()
    n_created = 0

    target_periods = critical_periods if critical_periods else periods

    for period in target_periods:
        if period not in periods:
            continue
        for band in band_list:
            band_col = f"{period}_{band}_mean"
            if band_col not in out.columns:
                continue
            for vi in vi_list:
                vi_col = f"{period}_{vi}_mean"
                if vi_col not in out.columns:
                    continue
                out[f"{period}_{band}_x_{vi}"] = out[band_col] * out[vi_col]
                n_created += 1

    logger.info("Band x VI interactions: %d features created", n_created)
    return out


# ---------------------------------------------------------------------------
# Bands x Soil interactions
# ---------------------------------------------------------------------------

def create_band_soil_interactions(
    df: pd.DataFrame,
    periods: List[str],
    band_list: List[str],
    soil_props: List[str],
    critical_periods: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create Band x Soil interaction features at critical periods."""
    out = df.copy()
    n_created = 0

    target_periods = critical_periods if critical_periods else periods

    for period in target_periods:
        if period not in periods:
            continue
        for band in band_list:
            band_col = f"{period}_{band}_mean"
            if band_col not in out.columns:
                continue
            for soil in soil_props:
                if soil not in out.columns:
                    continue
                out[f"{period}_{band}_x_{soil}"] = out[band_col] * out[soil]
                n_created += 1

    logger.info("Band x Soil interactions: %d features created", n_created)
    return out


# ---------------------------------------------------------------------------
# Bands x Meteo interactions
# ---------------------------------------------------------------------------

def create_band_meteo_interactions(
    df: pd.DataFrame,
    periods: List[str],
    band_list: List[str],
    meteo_vars: List[str],
    critical_periods: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create Band x Meteo interaction features at critical periods."""
    out = df.copy()
    n_created = 0

    target_periods = critical_periods if critical_periods else periods

    for period in target_periods:
        if period not in periods:
            continue
        for band in band_list:
            band_col = f"{period}_{band}_mean"
            if band_col not in out.columns:
                continue
            for meteo in meteo_vars:
                for suffix in ["_mean", "_sum", ""]:
                    meteo_col = f"{period}_{meteo}{suffix}"
                    if meteo_col in out.columns:
                        out[f"{period}_{band}_x_{meteo}"] = out[band_col] * out[meteo_col]
                        n_created += 1
                        break

    logger.info("Band x Meteo interactions: %d features created", n_created)
    return out


# ---------------------------------------------------------------------------
# Meteo x Soil interactions
# ---------------------------------------------------------------------------

def create_meteo_soil_interactions(
    df: pd.DataFrame,
    periods: List[str],
    meteo_vars: List[str],
    soil_props: List[str],
    critical_periods: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create Meteo x Soil interaction features at critical periods."""
    out = df.copy()
    n_created = 0

    target_periods = critical_periods if critical_periods else periods

    for period in target_periods:
        if period not in periods:
            continue
        for meteo in meteo_vars:
            meteo_col = None
            for suffix in ["_mean", "_sum", ""]:
                candidate = f"{period}_{meteo}{suffix}"
                if candidate in out.columns:
                    meteo_col = candidate
                    break
            if meteo_col is None:
                continue
            for soil in soil_props:
                if soil not in out.columns:
                    continue
                out[f"{period}_{meteo}_x_{soil}"] = out[meteo_col] * out[soil]
                n_created += 1

    logger.info("Meteo x Soil interactions: %d features created", n_created)
    return out


# ---------------------------------------------------------------------------
# VI x Soil interactions
# ---------------------------------------------------------------------------

def create_vi_soil_interactions(
    df: pd.DataFrame,
    periods: List[str],
    vi_list: List[str],
    soil_props: List[str],
    critical_periods: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create VI x Soil interaction features at critical periods."""
    out = df.copy()
    n_created = 0

    target_periods = critical_periods if critical_periods else periods

    for period in target_periods:
        if period not in periods:
            continue
        for vi in vi_list:
            vi_col = f"{period}_{vi}_mean"
            if vi_col not in out.columns:
                continue
            for soil in soil_props:
                if soil not in out.columns:
                    continue
                out[f"{period}_{vi}_x_{soil}"] = out[vi_col] * out[soil]
                n_created += 1

    logger.info("VI x Soil interactions: %d features created", n_created)
    return out


# ---------------------------------------------------------------------------
# Phenological features from double logistic params
# ---------------------------------------------------------------------------

def create_phenological_features(
    params_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create phenological features from double logistic params (m1..m7)."""
    if params_df.empty:
        return pd.DataFrame(columns=["field_key"])

    out = params_df[["field_key"]].copy()

    # Peak GCVI value (m1 + m2 at peak)
    out["pheno_baseline"] = params_df["m1"]
    out["pheno_amplitude"] = params_df["m2"]

    # Greenup steepness (smaller m4 = steeper greenup)
    out["pheno_greenup_steepness"] = params_df["m4"]

    # Senescence steepness (smaller m6 = steeper senescence)
    out["pheno_senescence_steepness"] = params_df["m6"]

    # Greenup inflection point
    out["pheno_greenup_inflection"] = params_df["m3"]

    # Senescence inflection point
    out["pheno_senescence_inflection"] = params_df["m5"]

    # Asymmetry: distance between greenup and senescence inflection points
    out["pheno_season_length"] = params_df["m5"] - params_df["m3"]

    # Trend parameter
    out["pheno_trend"] = params_df["m7"]

    # Greenup-to-senescence ratio (symmetry indicator)
    out["pheno_greenup_senescence_ratio"] = params_df["m4"] / (params_df["m6"] + EPSILON)

    logger.info("Phenological features: %d features from %d fields",
                len(out.columns) - 1, len(out))
    return out


# ---------------------------------------------------------------------------
# Seasonal aggregates
# ---------------------------------------------------------------------------

def create_seasonal_aggregates(
    df: pd.DataFrame,
    periods: List[str],
    feature_names: List[str],
) -> pd.DataFrame:
    """Create seasonal CV, range, and slope across periods (needs >= 3)."""
    if len(periods) < 3:
        logger.info("Seasonal aggregates: skipped (fewer than 3 periods)")
        return df

    out = df.copy()
    n_created = 0

    for feat in feature_names:
        cols = [f"{p}_{feat}_mean" for p in periods if f"{p}_{feat}_mean" in out.columns]
        if len(cols) < 3:
            continue

        values = out[cols].values  # (n_fields, n_periods)

        # CV
        means = np.nanmean(values, axis=1)
        stds = np.nanstd(values, axis=1, ddof=1)
        out[f"{feat}_seasonal_cv"] = stds / (np.abs(means) + EPSILON)
        n_created += 1

        # Range
        out[f"{feat}_seasonal_range"] = np.nanmax(values, axis=1) - np.nanmin(values, axis=1)
        n_created += 1

        # Slope (linear trend)
        x = np.arange(values.shape[1])
        slopes = []
        for row in values:
            valid = np.isfinite(row)
            if valid.sum() >= 2:
                slopes.append(np.polyfit(x[valid], row[valid], 1)[0])
            else:
                slopes.append(np.nan)
        out[f"{feat}_seasonal_slope"] = slopes
        n_created += 1

    logger.info("Seasonal aggregates: %d features created", n_created)
    return out


# ---------------------------------------------------------------------------
# Master dispatcher
# ---------------------------------------------------------------------------

def engineer_all_features(
    df: pd.DataFrame,
    periods: List[str],
    config: dict,
    params_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    fe_cfg = config["feature_engineering"]
    spec_cfg = config["spectral"]

    vi_names = spec_cfg["precomputed_indices"]
    band_names = spec_cfg["bands"]
    all_features = vi_names + band_names

    n_before = len(df.columns)

    # 1. Temporal derivatives
    if fe_cfg.get("temporal_derivatives", True):
        df = create_temporal_derivatives(df, periods, all_features)

    # 2. Band ratios
    df = create_band_ratios(df, periods, fe_cfg["band_ratios"])

    # 3. VI interactions
    df = create_vi_interactions(df, periods, fe_cfg["vi_interactions"])

    # 4. Agronomic indices
    df = create_agronomic_indices(df, periods)

    # 5. VI x Soil interactions
    vi_soil_cfg = fe_cfg["vi_soil_interactions"]
    # Use agronomically important periods for interactions:
    # - bw_peak: maximum biomass
    # - bw_p1, bw_p2: grain filling (critical for protein accumulation)
    # - bw_m1, bw_m2: heading/anthesis (nitrogen remobilization)
    # - bw_m3, bw_m4: stem elongation (rapid growth, N uptake)
    # Also support phenological stage names
    critical = [
        p for p in periods
        if any(x in p for x in [
            "peak", "p1", "p2",  # peak and post-peak
            "m1", "m2", "m3", "m4",  # pre-peak important periods
            "heading", "grain", "anthesis", "booting", "elongation"  # phenological
        ])
    ]
    if not critical:
        critical = periods[-3:] if len(periods) >= 3 else periods

    df = create_vi_soil_interactions(
        df, periods,
        vi_list=vi_soil_cfg["vis"],
        soil_props=vi_soil_cfg["soil_props"],
        critical_periods=critical,
    )

    # 6. VI x Meteo interactions
    if "vi_meteo_interactions" in fe_cfg:
        vi_meteo_cfg = fe_cfg["vi_meteo_interactions"]
        df = create_vi_meteo_interactions(
            df, periods,
            vi_list=vi_meteo_cfg["vis"],
            meteo_vars=vi_meteo_cfg["meteo_vars"],
            critical_periods=critical,
        )

    # 7. Band x VI interactions
    if "band_vi_interactions" in fe_cfg:
        band_vi_cfg = fe_cfg["band_vi_interactions"]
        df = create_band_vi_interactions(
            df, periods,
            band_list=band_vi_cfg["bands"],
            vi_list=band_vi_cfg["vis"],
            critical_periods=critical,
        )

    # 8. Band x Soil interactions
    if "band_soil_interactions" in fe_cfg:
        band_soil_cfg = fe_cfg["band_soil_interactions"]
        df = create_band_soil_interactions(
            df, periods,
            band_list=band_soil_cfg["bands"],
            soil_props=band_soil_cfg["soil_props"],
            critical_periods=critical,
        )

    # 9. Band x Meteo interactions
    if "band_meteo_interactions" in fe_cfg:
        band_meteo_cfg = fe_cfg["band_meteo_interactions"]
        df = create_band_meteo_interactions(
            df, periods,
            band_list=band_meteo_cfg["bands"],
            meteo_vars=band_meteo_cfg["meteo_vars"],
            critical_periods=critical,
        )

    # 10. Meteo x Soil interactions
    if "meteo_soil_interactions" in fe_cfg:
        meteo_soil_cfg = fe_cfg["meteo_soil_interactions"]
        df = create_meteo_soil_interactions(
            df, periods,
            meteo_vars=meteo_soil_cfg["meteo_vars"],
            soil_props=meteo_soil_cfg["soil_props"],
            critical_periods=critical,
        )

    # 11. Phenological features
    if fe_cfg.get("phenological_features", True) and params_df is not None:
        pheno_df = create_phenological_features(params_df)
        if not pheno_df.empty:
            df = df.merge(pheno_df, on="field_key", how="left")

    # 12. Seasonal aggregates
    df = create_seasonal_aggregates(df, periods, vi_names[:10])

    n_after = len(df.columns)
    logger.info(
        "Feature engineering complete: %d → %d columns (+%d)",
        n_before, n_after, n_after - n_before,
    )
    return df
