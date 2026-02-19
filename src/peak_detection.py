"""GCVI peak detection pipeline: SG smoothing + double logistic fit + QC."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .double_logistic import (
    double_logistic,
    find_peak_fallback,
    find_peak_from_params,
    fit_double_logistic,
)
from .smoothing import savgol_smooth_gcvi

logger = logging.getLogger("gpc_pipeline")


def detect_peaks(
    spectral_df: pd.DataFrame,
    config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run full GCVI peak detection: SG smoothing → double logistic fit → QC flagging."""
    pd_cfg = config["peak_detection"]
    qc_cfg = config["peak_qc"]

    ref_var = pd_cfg["reference_var"]
    fit_start = pd.Timestamp(pd_cfg["fit_start_date"])
    fit_end = pd.Timestamp(pd_cfg["fit_end_date"])
    resample_days = pd_cfg["resample_days"]
    sg_window = pd_cfg["savgol_window"]
    sg_poly = pd_cfg["savgol_polyorder"]
    interpolate_daily = pd_cfg.get("interpolate_daily", True)
    max_nfev = pd_cfg["lmfit_max_nfev"]
    valid_start_month = pd_cfg["valid_peak_start_month"]
    valid_end_month = pd_cfg["valid_peak_end_month"]

    # Additional QC thresholds for curve quality
    min_peak_gcvi = qc_cfg.get("min_peak_gcvi", 0)
    min_amplitude = qc_cfg.get("min_amplitude", 0)
    min_fit_r2 = qc_cfg.get("min_fit_r2", 0)

    # Step 1: Filter to fitting date range
    mask = (spectral_df["date"] >= fit_start) & (spectral_df["date"] <= fit_end)
    fit_df = spectral_df.loc[mask].copy()
    logger.info(
        "Peak detection: %d rows in fitting window (%s to %s)",
        len(fit_df),
        fit_start.date(),
        fit_end.date(),
    )

    if fit_df.empty:
        raise ValueError("No spectral data in the fitting window")

    # Step 2: SG smoothing (with optional daily interpolation)
    profiles = savgol_smooth_gcvi(
        fit_df,
        reference_var=ref_var,
        resample_days=resample_days,
        savgol_window=sg_window,
        savgol_polyorder=sg_poly,
        interpolate_daily=interpolate_daily,
    )

    # Step 3 & 4: Double logistic fit + peak detection
    results = []
    params_records = []
    fit_success = 0
    fit_fallback = 0

    for fk, (dates, smoothed) in profiles.items():
        # Convert dates to DOY for fitting
        dates_ts = pd.DatetimeIndex(dates)
        doy = dates_ts.dayofyear.values.astype(float)

        # Try double logistic fit
        params = fit_double_logistic(doy, smoothed, max_nfev=max_nfev)

        # Initialize QC metrics
        fit_r2 = None
        amplitude = None

        if params is not None:
            peak_doy, peak_gcvi = find_peak_from_params(
                params,
                (doy.min(), doy.max()),
            )
            method = "double_logistic"
            fit_success += 1

            # Extract QC metrics
            fit_r2 = params.get("fit_r2", None)
            amplitude = params.get("m2", None)

            # Store fitted parameters
            params_record = {"field_key": fk}
            params_record.update(params)
            params_records.append(params_record)
        else:
            peak_doy, peak_gcvi = find_peak_fallback(doy, smoothed)
            method = "fallback_rolling_mean"
            fit_fallback += 1

        # Convert DOY back to date
        year = fit_start.year
        try:
            peak_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(
                days=int(peak_doy) - 1,
            )
        except (ValueError, OverflowError):
            peak_date = pd.NaT

        peak_month = peak_date.month if pd.notna(peak_date) else None

        # QC: flag anomalous peaks (multiple criteria)
        anomalous = False
        anomaly_reasons = []

        # Check 1: Peak month outside valid range
        if peak_month is None:
            anomalous = True
            anomaly_reasons.append("no_peak_date")
        elif not (valid_start_month <= peak_month <= valid_end_month):
            anomalous = True
            anomaly_reasons.append("peak_outside_season")

        # Check 2: Peak GCVI too low (non-vegetated or sparse)
        if min_peak_gcvi > 0 and peak_gcvi < min_peak_gcvi:
            anomalous = True
            anomaly_reasons.append("low_peak_gcvi")

        # Check 3: Amplitude too low (flat curve, no bell shape)
        if min_amplitude > 0 and amplitude is not None and amplitude < min_amplitude:
            anomalous = True
            anomaly_reasons.append("low_amplitude")

        # Check 4: Poor fit quality
        if min_fit_r2 > 0 and fit_r2 is not None and fit_r2 < min_fit_r2:
            anomalous = True
            anomaly_reasons.append("poor_fit")

        results.append(
            {
                "field_key": fk,
                "peak_date": peak_date,
                "peak_doy": peak_doy,
                "peak_gcvi": peak_gcvi,
                "fit_method": method,
                "peak_anomalous": anomalous,
                "anomaly_reason": (
                    ";".join(anomaly_reasons) if anomaly_reasons else None
                ),
                "peak_month": peak_month,
                "fit_r2": fit_r2,
                "amplitude": amplitude,
            }
        )

    skipped = len(profiles) - fit_success - fit_fallback
    n_fields_total = spectral_df["field_key"].nunique()
    logger.info(
        "Peak detection complete: %d/%d fields — "
        "%d double logistic, %d fallback, %d skipped in smoothing",
        len(results),
        n_fields_total,
        fit_success,
        fit_fallback,
        n_fields_total - len(profiles),
    )

    peak_df = pd.DataFrame(results)
    params_df = pd.DataFrame(params_records) if params_records else pd.DataFrame()

    # Log QC summary
    if not peak_df.empty:
        n_anomalous = peak_df["peak_anomalous"].sum()
        pct_anomalous = 100 * n_anomalous / len(peak_df)
        logger.info(
            "Peak QC: %d/%d (%.1f%%) fields flagged as anomalous",
            n_anomalous,
            len(peak_df),
            pct_anomalous,
        )
        # Log breakdown by anomaly reason
        if n_anomalous > 0:
            reasons = peak_df.loc[peak_df["peak_anomalous"], "anomaly_reason"]
            reason_counts = {}
            for r in reasons.dropna():
                for reason in r.split(";"):
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
            for reason, count in sorted(reason_counts.items()):
                logger.info("  - %s: %d fields", reason, count)

    return peak_df, params_df


def peak_qc_report(peak_df: pd.DataFrame, config: dict) -> Dict:
    """Generate a QC report for peak detection results."""
    valid_start = config["peak_detection"]["valid_peak_start_month"]
    valid_end = config["peak_detection"]["valid_peak_end_month"]
    qc_cfg = config["peak_qc"]

    total = len(peak_df)
    n_anomalous = int(peak_df["peak_anomalous"].sum())
    n_valid = total - n_anomalous
    pct_anomalous = 100 * n_anomalous / total if total > 0 else 0

    # Peak month distribution
    month_dist = peak_df["peak_month"].value_counts().sort_index().to_dict()

    # Fit method distribution
    method_dist = peak_df["fit_method"].value_counts().to_dict()

    # Peak DOY statistics
    doy_stats = {
        "mean": float(peak_df["peak_doy"].mean()),
        "median": float(peak_df["peak_doy"].median()),
        "std": float(peak_df["peak_doy"].std()),
        "min": float(peak_df["peak_doy"].min()),
        "max": float(peak_df["peak_doy"].max()),
    }

    # Anomaly reason breakdown
    reason_counts = {}
    if n_anomalous > 0 and "anomaly_reason" in peak_df.columns:
        reasons = peak_df.loc[peak_df["peak_anomalous"], "anomaly_reason"]
        for r in reasons.dropna():
            for reason in r.split(";"):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

    # Fit quality stats
    fit_r2_stats = {}
    if "fit_r2" in peak_df.columns:
        valid_r2 = peak_df["fit_r2"].dropna()
        if len(valid_r2) > 0:
            fit_r2_stats = {
                "mean": float(valid_r2.mean()),
                "median": float(valid_r2.median()),
                "min": float(valid_r2.min()),
                "max": float(valid_r2.max()),
            }

    report = {
        "total_fields": total,
        "valid_fields": n_valid,
        "anomalous_fields": n_anomalous,
        "pct_anomalous": round(pct_anomalous, 2),
        "valid_peak_range": f"Month {valid_start} to {valid_end}",
        "qc_thresholds": {
            "min_peak_gcvi": qc_cfg.get("min_peak_gcvi", 0),
            "min_amplitude": qc_cfg.get("min_amplitude", 0),
            "min_fit_r2": qc_cfg.get("min_fit_r2", 0),
        },
        "anomaly_reasons": reason_counts,
        "peak_month_distribution": month_dist,
        "fit_method_distribution": method_dist,
        "peak_doy_stats": doy_stats,
        "fit_r2_stats": fit_r2_stats,
    }

    return report


def get_smoothed_and_fitted_curves(
    spectral_df: pd.DataFrame,
    peak_df: pd.DataFrame,
    params_df: pd.DataFrame,
    config: dict,
    field_key: str,
) -> Dict:
    """Get smoothed and fitted curve data for a single field (for plotting)."""
    pd_cfg = config["peak_detection"]
    ref_var = pd_cfg["reference_var"]
    fit_start = pd.Timestamp(pd_cfg["fit_start_date"])
    fit_end = pd.Timestamp(pd_cfg["fit_end_date"])

    # Raw data
    mask = (
        (spectral_df["field_key"] == field_key)
        & (spectral_df["date"] >= fit_start)
        & (spectral_df["date"] <= fit_end)
    )
    raw = spectral_df.loc[mask, ["date", ref_var]].dropna()

    # Smoothed profile
    profiles = savgol_smooth_gcvi(
        spectral_df.loc[mask],
        reference_var=ref_var,
        resample_days=pd_cfg["resample_days"],
        savgol_window=pd_cfg["savgol_window"],
        savgol_polyorder=pd_cfg["savgol_polyorder"],
        interpolate_daily=pd_cfg.get("interpolate_daily", True),
    )

    result = {
        "raw_dates": raw["date"].values,
        "raw_gcvi": raw[ref_var].values,
    }

    if field_key in profiles:
        dates, smoothed = profiles[field_key]
        result["smooth_dates"] = dates
        result["smooth_gcvi"] = smoothed

    # Fitted curve
    if not params_df.empty and field_key in params_df["field_key"].values:
        row = params_df[params_df["field_key"] == field_key].iloc[0]
        params = {f"m{i}": row[f"m{i}"] for i in range(1, 8)}
        doy_range = np.linspace(fit_start.dayofyear, fit_end.dayofyear, 500)
        fitted = double_logistic(
            doy_range,
            params["m1"],
            params["m2"],
            params["m3"],
            params["m4"],
            params["m5"],
            params["m6"],
            params["m7"],
        )
        result["fitted_doy"] = doy_range
        result["fitted_gcvi"] = fitted

    # Peak info
    peak_row = peak_df[peak_df["field_key"] == field_key]
    if not peak_row.empty:
        result["peak_date"] = peak_row.iloc[0]["peak_date"]
        result["peak_gcvi"] = peak_row.iloc[0]["peak_gcvi"]

    return result


# ---------------------------------------------------------------------------
# Peak QC Visualization
# ---------------------------------------------------------------------------


def plot_peak_qc_dashboard(
    peak_df: pd.DataFrame,
    config: dict,
    figsize: tuple = (14, 10),
    save_path: Optional[str] = None,
):
    """4-panel QC dashboard for peak detection results."""
    import matplotlib.pyplot as plt

    qc_cfg = config["peak_qc"]
    min_peak_gcvi = qc_cfg.get("min_peak_gcvi", 0)
    min_amplitude = qc_cfg.get("min_amplitude", 0)
    min_fit_r2 = qc_cfg.get("min_fit_r2", 0)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Peak Detection QC Dashboard", fontsize=14, fontweight="bold")

    # Panel 1: Anomaly reasons bar chart
    ax1 = axes[0, 0]
    _plot_anomaly_reasons(peak_df, ax=ax1)

    # Panel 2: Peak GCVI distribution
    ax2 = axes[0, 1]
    _plot_metric_distribution(
        peak_df,
        "peak_gcvi",
        min_peak_gcvi,
        "Peak GCVI Distribution",
        "Peak GCVI",
        ax=ax2,
    )

    # Panel 3: Amplitude distribution
    ax3 = axes[1, 0]
    _plot_metric_distribution(
        peak_df,
        "amplitude",
        min_amplitude,
        "Amplitude (m2) Distribution",
        "Amplitude",
        ax=ax3,
    )

    # Panel 4: Fit R² distribution
    ax4 = axes[1, 1]
    _plot_metric_distribution(
        peak_df,
        "fit_r2",
        min_fit_r2,
        "Fit R² Distribution",
        "R²",
        ax=ax4,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved QC dashboard to %s", save_path)

    return fig


def _plot_anomaly_reasons(peak_df: pd.DataFrame, ax=None):
    """Plot bar chart of anomaly reasons."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Count anomaly reasons
    reason_counts = {}
    if "anomaly_reason" in peak_df.columns:
        reasons = peak_df.loc[peak_df["peak_anomalous"], "anomaly_reason"]
        for r in reasons.dropna():
            for reason in r.split(";"):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

    n_valid = (~peak_df["peak_anomalous"]).sum()
    n_anomalous = peak_df["peak_anomalous"].sum()

    # Add valid count for comparison
    all_counts = {"valid": n_valid, **reason_counts}

    # Sort by count
    sorted_items = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    # Colors
    colors = ["#2ecc71" if label == "valid" else "#e74c3c" for label in labels]

    bars = ax.barh(labels, values, color=colors, edgecolor="white")
    ax.set_xlabel("Number of Fields")
    ax.set_title(f"Field Status (Valid: {n_valid}, Anomalous: {n_anomalous})")

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center",
            fontsize=9,
        )

    ax.invert_yaxis()
    return ax


def _plot_metric_distribution(
    peak_df: pd.DataFrame,
    column: str,
    threshold: float,
    title: str,
    xlabel: str,
    ax=None,
):
    """Plot histogram of a QC metric with threshold line."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    if column not in peak_df.columns:
        ax.text(
            0.5,
            0.5,
            f"Column '{column}' not found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        return ax

    data = peak_df[column].dropna()
    if len(data) == 0:
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        return ax

    # Split by anomalous status
    valid_mask = ~peak_df["peak_anomalous"]
    valid_data = peak_df.loc[valid_mask, column].dropna()
    anomalous_data = peak_df.loc[~valid_mask, column].dropna()

    # Plot histograms
    bins = 30
    ax.hist(
        valid_data,
        bins=bins,
        alpha=0.7,
        label="Valid",
        color="#2ecc71",
        edgecolor="white",
    )
    ax.hist(
        anomalous_data,
        bins=bins,
        alpha=0.7,
        label="Anomalous",
        color="#e74c3c",
        edgecolor="white",
    )

    # Add threshold line
    if threshold > 0:
        ax.axvline(
            threshold,
            color="#3498db",
            linestyle="--",
            linewidth=2,
            label=f"Threshold: {threshold}",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(loc="upper right")

    # Add stats text
    stats_text = (
        f"Mean: {data.mean():.3f}\nMedian: {data.median():.3f}\nMin: {data.min():.3f}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    return ax


def plot_qc_scatter(
    peak_df: pd.DataFrame,
    config: dict,
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
):
    """Scatter plots showing relationships between QC metrics."""
    import matplotlib.pyplot as plt

    qc_cfg = config["peak_qc"]
    min_peak_gcvi = qc_cfg.get("min_peak_gcvi", 0)
    min_amplitude = qc_cfg.get("min_amplitude", 0)
    min_fit_r2 = qc_cfg.get("min_fit_r2", 0)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle("QC Metrics Relationships", fontsize=12, fontweight="bold")

    # Colors based on anomalous status
    colors = peak_df["peak_anomalous"].map({True: "#e74c3c", False: "#2ecc71"})

    # Panel 1: Peak GCVI vs Amplitude
    ax1 = axes[0]
    ax1.scatter(
        peak_df["peak_gcvi"],
        peak_df["amplitude"],
        c=colors,
        alpha=0.6,
        edgecolors="white",
        s=40,
    )
    if min_peak_gcvi > 0:
        ax1.axvline(min_peak_gcvi, color="#3498db", linestyle="--", alpha=0.7)
    if min_amplitude > 0:
        ax1.axhline(min_amplitude, color="#3498db", linestyle="--", alpha=0.7)
    ax1.set_xlabel("Peak GCVI")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Peak GCVI vs Amplitude")

    # Panel 2: Peak GCVI vs Fit R²
    ax2 = axes[1]
    ax2.scatter(
        peak_df["peak_gcvi"],
        peak_df["fit_r2"],
        c=colors,
        alpha=0.6,
        edgecolors="white",
        s=40,
    )
    if min_peak_gcvi > 0:
        ax2.axvline(min_peak_gcvi, color="#3498db", linestyle="--", alpha=0.7)
    if min_fit_r2 > 0:
        ax2.axhline(min_fit_r2, color="#3498db", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Peak GCVI")
    ax2.set_ylabel("Fit R²")
    ax2.set_title("Peak GCVI vs Fit R²")

    # Panel 3: Amplitude vs Fit R²
    ax3 = axes[2]
    sc = ax3.scatter(
        peak_df["amplitude"],
        peak_df["fit_r2"],
        c=colors,
        alpha=0.6,
        edgecolors="white",
        s=40,
    )
    if min_amplitude > 0:
        ax3.axvline(min_amplitude, color="#3498db", linestyle="--", alpha=0.7)
    if min_fit_r2 > 0:
        ax3.axhline(min_fit_r2, color="#3498db", linestyle="--", alpha=0.7)
    ax3.set_xlabel("Amplitude")
    ax3.set_ylabel("Fit R²")
    ax3.set_title("Amplitude vs Fit R²")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2ecc71", label="Valid"),
        Patch(facecolor="#e74c3c", label="Anomalous"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.99, 0.99))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved QC scatter plot to %s", save_path)

    return fig


def plot_anomalous_fields_grid(
    spectral_df: pd.DataFrame,
    peak_df: pd.DataFrame,
    params_df: pd.DataFrame,
    config: dict,
    max_fields: int = 12,
    figsize: tuple = (15, 12),
    save_path: Optional[str] = None,
):
    """Plot GCVI curves for anomalous fields in a grid."""
    import matplotlib.pyplot as plt

    anomalous_df = peak_df[peak_df["peak_anomalous"]].copy()
    if len(anomalous_df) == 0:
        logger.info("No anomalous fields to plot")
        return None

    n_fields = min(len(anomalous_df), max_fields)
    n_cols = 4
    n_rows = (n_fields + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(
        f"Anomalous Fields ({len(anomalous_df)} total, showing {n_fields})",
        fontsize=12,
        fontweight="bold",
    )

    axes_flat = axes.flatten() if n_fields > 1 else [axes]

    for idx, (_, row) in enumerate(anomalous_df.head(max_fields).iterrows()):
        ax = axes_flat[idx]
        fk = row["field_key"]

        try:
            curves = get_smoothed_and_fitted_curves(
                spectral_df,
                peak_df,
                params_df,
                config,
                fk,
            )

            # Plot raw data
            if "raw_dates" in curves:
                ax.scatter(
                    curves["raw_dates"],
                    curves["raw_gcvi"],
                    s=20,
                    alpha=0.5,
                    label="Raw",
                    color="#3498db",
                )

            # Plot smoothed
            if "smooth_dates" in curves:
                ax.plot(
                    curves["smooth_dates"],
                    curves["smooth_gcvi"],
                    linewidth=1.5,
                    label="Smoothed",
                    color="#2ecc71",
                )

            # Plot fitted curve
            if "fitted_doy" in curves:
                # Convert DOY to dates for plotting
                year = pd.Timestamp(config["peak_detection"]["fit_start_date"]).year
                fitted_dates = [
                    pd.Timestamp(year=year, month=1, day=1)
                    + pd.Timedelta(days=int(d) - 1)
                    for d in curves["fitted_doy"]
                ]
                ax.plot(
                    fitted_dates,
                    curves["fitted_gcvi"],
                    linewidth=1.5,
                    linestyle="--",
                    label="Fitted",
                    color="#e74c3c",
                )

            # Mark peak
            if "peak_date" in curves and pd.notna(curves["peak_date"]):
                ax.axvline(
                    curves["peak_date"], color="#9b59b6", linestyle=":", alpha=0.7
                )

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error: {e}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )

        # Title with anomaly reason
        reason = row.get("anomaly_reason", "unknown")
        title = f"{fk[:20]}...\n{reason}" if len(fk) > 20 else f"{fk}\n{reason}"
        ax.set_title(title, fontsize=8)
        ax.tick_params(axis="both", labelsize=7)

    # Hide unused axes
    for idx in range(n_fields, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved anomalous fields grid to %s", save_path)

    return fig
