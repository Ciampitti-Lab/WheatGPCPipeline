"""Savitzky-Golay smoothing for GCVI time-series profiles."""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

logger = logging.getLogger("gpc_pipeline")


def savgol_smooth_gcvi(
    spectral_df: pd.DataFrame,
    reference_var: str = "GCVI",
    resample_days: int = 3,
    savgol_window: int = 7,
    savgol_polyorder: int = 2,
    interpolate_daily: bool = True,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """SG-smooth GCVI profiles per field. Returns {field_key: (dates, values)}."""
    if reference_var not in spectral_df.columns:
        raise ValueError(
            f"Reference variable '{reference_var}' not in spectral data. "
            f"Available: {[c for c in spectral_df.columns if c not in ('field_key', 'date')]}"
        )

    # Step 1: Mean reference_var per field per date
    grouped = (
        spectral_df.groupby(["field_key", "date"])[reference_var]
        .mean()
        .reset_index()
    )

    # Step 2: Build regular time grid
    fields = grouped["field_key"].unique()
    date_min = grouped["date"].min()
    date_max = grouped["date"].max()

    # Create daily grid for interpolation, then resample
    daily_dates = pd.date_range(date_min, date_max, freq="1D")
    output_dates = pd.date_range(date_min, date_max, freq=f"{resample_days}D")

    profiles = {}

    for fk in fields:
        sub = grouped.loc[grouped["field_key"] == fk].set_index("date")[reference_var]

        # Skip fields with too few observations
        n_valid = sub.dropna().shape[0]
        if n_valid < 3:
            logger.debug("Field %s: only %d observations, skipping smoothing", fk, n_valid)
            continue

        if interpolate_daily:
            # First interpolate to daily grid for denser data
            sub_daily = sub.reindex(daily_dates).interpolate(
                method="linear", limit_direction="both",
            )
            sub_daily = sub_daily.ffill().bfill()

            # Apply SG smoothing on daily data (use larger window for daily)
            daily_win = min(savgol_window * resample_days, len(sub_daily))
            if daily_win % 2 == 0:
                daily_win -= 1
            if daily_win < savgol_polyorder + 2:
                smoothed_daily = sub_daily.values
            else:
                smoothed_daily = savgol_filter(
                    sub_daily.values, window_length=daily_win, polyorder=savgol_polyorder
                )

            # Then resample to output grid
            smoothed_series = pd.Series(smoothed_daily, index=daily_dates)
            smoothed = smoothed_series.reindex(output_dates, method="nearest").values
        else:
            # Original behavior: resample directly
            sub = sub.reindex(output_dates).interpolate(
                method="linear", limit_direction="both",
            )
            sub = sub.ffill().bfill()
            vals = sub.values

            win = min(savgol_window, len(vals))
            if win % 2 == 0:
                win -= 1
            if win < savgol_polyorder + 2:
                smoothed = vals
            else:
                smoothed = savgol_filter(vals, window_length=win, polyorder=savgol_polyorder)

        profiles[fk] = (output_dates.values, smoothed)

    logger.info(
        "SG smoothing: %d/%d fields processed (resample=%dd, SG(%d,%d), daily_interp=%s)",
        len(profiles), len(fields), resample_days, savgol_window, savgol_polyorder,
        interpolate_daily,
    )

    return profiles
