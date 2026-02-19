"""Temporal alignment and aggregation with 3 modes: biweekly, monthly, phenological."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import get_feature_type

logger = logging.getLogger("gpc_pipeline")

EPSILON = 1e-6


# ---------------------------------------------------------------------------
# Time normalization
# ---------------------------------------------------------------------------


def normalize_to_peak(
    df: pd.DataFrame,
    peak_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert dates to peak-relative days (peak = day 0)."""
    peak_map = peak_df.set_index("field_key")["peak_date"]

    out = df.copy()
    out["peak_date"] = out["field_key"].map(peak_map)

    before = len(out)
    out = out.dropna(subset=["peak_date"])
    dropped = before - len(out)
    if dropped > 0:
        logger.info("Dropped %d rows (fields without peak date)", dropped)

    out["peak_date"] = pd.to_datetime(out["peak_date"])
    out["normalized_day"] = (out["date"] - out["peak_date"]).dt.days
    out.drop(columns=["peak_date"], inplace=True)

    logger.info(
        "Time normalization: %d fields, normalized_day range [%d, %d]",
        out["field_key"].nunique(),
        int(out["normalized_day"].min()),
        int(out["normalized_day"].max()),
    )
    return out


# ---------------------------------------------------------------------------
# Period window generation
# ---------------------------------------------------------------------------


def generate_biweekly_windows(config: dict) -> List[Tuple[str, int, int]]:
    """Generate bi-weekly windows centered on peak."""
    bw_cfg = config["aggregation"]["biweekly"]
    peak_start, peak_end = bw_cfg["peak_window"]
    step = bw_cfg["step"]
    n_before = bw_cfg["n_before"]
    n_after = bw_cfg["n_after"]

    windows = []

    # Before peak
    for i in range(n_before, 0, -1):
        w_end = peak_start - (i - 1) * step
        w_start = w_end - step
        windows.append((f"bw_m{i}", w_start, w_end - 1))

    # Peak biweek
    windows.append(("bw_peak", peak_start, peak_end))

    # After peak
    for i in range(1, n_after + 1):
        w_start = peak_end + 1 + (i - 1) * step
        w_end = w_start + step - 1
        windows.append((f"bw_p{i}", w_start, w_end))

    return windows


def generate_monthly_windows(config: dict) -> List[Tuple[str, int, int]]:
    """Generate monthly windows centered on peak."""
    mo_cfg = config["aggregation"]["monthly"]
    peak_start, peak_end = mo_cfg["peak_window"]
    step = mo_cfg["step"]
    n_before = mo_cfg["n_before"]
    n_after = mo_cfg["n_after"]

    windows = []

    # Before peak
    for i in range(n_before, 0, -1):
        w_end = peak_start - 1 - (i - 1) * step
        w_start = w_end - step + 1
        windows.append((f"mo_m{i}", w_start, w_end))

    # Peak month
    windows.append(("mo_peak", peak_start, peak_end))

    # After peak
    for i in range(1, n_after + 1):
        w_start = peak_end + 1 + (i - 1) * step
        w_end = w_start + step - 1
        windows.append((f"mo_p{i}", w_start, w_end))

    return windows


def generate_phenological_windows(config: dict) -> List[Tuple[str, int, int]]:
    """Generate phenological stage windows from config."""
    stages = config["aggregation"]["phenological"]["stages"]
    windows = []
    for name, (start, end) in stages.items():
        windows.append((name, start, end))
    return windows


def generate_peak_stages_windows(config: dict) -> List[Tuple[str, int, int]]:
    """Generate peak-relative custom stage windows from config."""
    stages = config["aggregation"]["peak_stages"]["stages"]
    windows = []
    for name, (start, end) in stages.items():
        windows.append((name, start, end))
    return windows


def _calendar_stages_to_windows(
    stages: dict,
) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Convert a dict of {name: [start_date, end_date]} to Timestamp windows."""
    windows = []
    for name, (start, end) in stages.items():
        windows.append((name, pd.Timestamp(start), pd.Timestamp(end)))
    return windows


def generate_custom_windows(
    config: dict,
) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Generate custom calendar-based period windows from config."""
    return _calendar_stages_to_windows(config["aggregation"]["custom"]["stages"])


def generate_calendar_monthly_windows(
    config: dict,
) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Generate calendar month windows (Jul–Jun) from config."""
    return _calendar_stages_to_windows(
        config["aggregation"]["calendar_monthly"]["stages"]
    )


def generate_calendar_biweekly_windows(
    config: dict,
) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Generate calendar biweekly windows (Jul–Jun, 24 periods) from config."""
    return _calendar_stages_to_windows(
        config["aggregation"]["calendar_biweekly"]["stages"]
    )


def generate_period_windows(config: dict) -> List[Tuple[str, int, int]]:
    """Generate period windows based on config aggregation mode."""
    mode = config["aggregation"]["mode"]
    if mode == "biweekly":
        return generate_biweekly_windows(config)
    elif mode == "monthly":
        return generate_monthly_windows(config)
    elif mode == "phenological":
        return generate_phenological_windows(config)
    elif mode == "peak_stages":
        return generate_peak_stages_windows(config)
    elif mode == "custom":
        return generate_custom_windows(config)
    elif mode == "calendar_monthly":
        return generate_calendar_monthly_windows(config)
    elif mode == "calendar_biweekly":
        return generate_calendar_biweekly_windows(config)
    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")


# ---------------------------------------------------------------------------
# Aggregation functions
# ---------------------------------------------------------------------------


def _compute_agg(values: np.ndarray, stat: str) -> float:
    """Compute a single aggregation statistic on an array."""
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan

    if stat == "mean":
        return float(np.mean(values))
    elif stat == "std":
        return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    elif stat == "p10":
        return float(np.percentile(values, 10))
    elif stat == "p90":
        return float(np.percentile(values, 90))
    elif stat == "sum":
        return float(np.sum(values))
    elif stat == "min":
        return float(np.min(values))
    elif stat == "max":
        return float(np.max(values))
    elif stat == "range":
        return float(np.max(values) - np.min(values))
    elif stat == "cv":
        m = np.mean(values)
        return (
            float(np.std(values, ddof=1) / (abs(m) + EPSILON))
            if len(values) > 1
            else 0.0
        )
    elif stat == "slope":
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        return float(np.polyfit(x, values, 1)[0])
    else:
        raise ValueError(f"Unknown aggregation stat: {stat}")


def aggregate_periods(
    spectral_norm: pd.DataFrame,
    meteo_norm: pd.DataFrame,
    config: dict,
    feature_columns: Optional[List[str]] = None,
    meteo_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Aggregate spectral and meteo data into temporal periods."""
    windows = generate_period_windows(config)
    agg_cfg = config["aggregation"]["agg_functions"]
    use_dates = config["aggregation"]["mode"] in (
        "custom",
        "calendar_monthly",
        "calendar_biweekly",
    )
    filter_col = "date" if use_dates else "normalized_day"

    # Auto-detect feature columns
    exclude_cols = {"field_key", "date", "normalized_day", "peak_date"}
    if feature_columns is None:
        feature_columns = [c for c in spectral_norm.columns if c not in exclude_cols]
    if meteo_columns is None:
        meteo_columns = [c for c in meteo_norm.columns if c not in exclude_cols]

    logger.info(
        "Aggregating %d spectral + %d meteo features across %d periods",
        len(feature_columns),
        len(meteo_columns),
        len(windows),
    )

    fields = sorted(
        set(spectral_norm["field_key"].unique()) & set(meteo_norm["field_key"].unique())
    )

    all_records = []

    for fk in fields:
        record = {"field_key": fk}
        spec_fk = spectral_norm[spectral_norm["field_key"] == fk]
        met_fk = meteo_norm[meteo_norm["field_key"] == fk]

        for period_name, start_val, end_val in windows:
            # Spectral aggregation
            spec_window = spec_fk[
                (spec_fk[filter_col] >= start_val) & (spec_fk[filter_col] <= end_val)
            ]

            for feat in feature_columns:
                if feat not in spec_window.columns:
                    continue
                feat_type = get_feature_type(feat)
                stats = agg_cfg.get(feat_type, ["mean"])

                values = spec_window[feat].values
                for stat in stats:
                    col_name = f"{period_name}_{feat}_{stat}"
                    record[col_name] = _compute_agg(values, stat)

            # Meteo aggregation
            met_window = met_fk[
                (met_fk[filter_col] >= start_val) & (met_fk[filter_col] <= end_val)
            ]

            for feat in meteo_columns:
                if feat not in met_window.columns:
                    continue
                feat_type = get_feature_type(feat)
                stats = agg_cfg.get(feat_type, ["mean"])

                values = met_window[feat].values
                for stat in stats:
                    col_name = f"{period_name}_{feat}_{stat}"
                    record[col_name] = _compute_agg(values, stat)

        all_records.append(record)

    result = pd.DataFrame(all_records)
    logger.info(
        "Aggregation complete: %d fields x %d columns",
        len(result),
        len(result.columns) - 1,
    )
    return result


def compute_derived_meteo(
    meteo_norm: pd.DataFrame,
    gdd_base_temp: float = 0.0,
) -> pd.DataFrame:
    """Compute VPD, GDD, and water deficit from meteo data."""
    out = meteo_norm.copy()

    # VPD (Tetens formula)
    if "t2m_mean" in out.columns and "dewpoint" in out.columns:
        es = 0.6108 * np.exp(17.27 * out["t2m_mean"] / (out["t2m_mean"] + 237.3))
        ea = 0.6108 * np.exp(17.27 * out["dewpoint"] / (out["dewpoint"] + 237.3))
        out["vpd"] = (es - ea).clip(lower=0)

    # GDD
    if "t2m_max" in out.columns and "t2m_min" in out.columns:
        t_avg = (out["t2m_max"] + out["t2m_min"]) / 2
        out["gdd"] = (t_avg - gdd_base_temp).clip(lower=0)

    # Water deficit
    if "pet" in out.columns and "precip" in out.columns:
        out["water_deficit"] = out["pet"] - out["precip"]

    return out
