"""Evaluation metrics and visualization for the GPC Pipeline."""

import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger("gpc_pipeline")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Lin's Concordance Correlation Coefficient."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mean_t, mean_p = y_true.mean(), y_pred.mean()
    var_t, var_p = y_true.var(), y_pred.var()
    sd_t, sd_p = y_true.std(), y_pred.std()
    rho = np.corrcoef(y_true, y_pred)[0, 1]
    numerator = 2 * rho * sd_t * sd_p
    denominator = var_t + var_p + (mean_t - mean_p) ** 2
    return float(numerator / (denominator + 1e-10))


def kge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Kling-Gupta Efficiency."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = y_pred.std() / (y_true.std() + 1e-10)
    beta = y_pred.mean() / (y_true.mean() + 1e-10)
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def pla_plp(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """PLA and PLP (systematic bias vs random scatter)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    mean_obs = y_true.mean()

    coeffs = np.polyfit(y_true, y_pred, 1)
    y_pred_fitted = np.polyval(coeffs, y_true)

    pla_val = 100 * np.sqrt(np.sum((y_pred_fitted - y_true) ** 2) / n) / (mean_obs + 1e-10)
    plp_val = 100 * np.sqrt(np.sum((y_pred - y_pred_fitted) ** 2) / n) / (mean_obs + 1e-10)

    return float(pla_val), float(plp_val)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute R2, CCC, RMSE, RRMSE, MBE, PLA, PLP, KGE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_obs = y_true.mean()
    pla_val, plp_val = pla_plp(y_true, y_pred)

    return {
        "R2": float(r2_score(y_true, y_pred)),
        "CCC": ccc(y_true, y_pred),
        "RMSE": float(rmse),
        "RRMSE": float(100 * rmse / (mean_obs + 1e-10)),
        "MBE": float(np.mean(y_pred - y_true)),
        "PLA": pla_val,
        "PLP": plp_val,
        "KGE": kge(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_predictions_vs_obs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
    ax: Optional[plt.Axes] = None,
    metrics: Optional[Dict[str, float]] = None,
    prediction_interval: Optional[str] = "rmse",
) -> plt.Axes:
    """Plot predicted vs observed with 1:1 line and optional prediction interval."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Calculate RMSE for interval bands
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Define plot limits
    lims = [
        min(min(y_true), min(y_pred)) - 0.5,
        max(max(y_true), max(y_pred)) + 0.5,
    ]
    x_line = np.linspace(lims[0], lims[1], 100)

    # Draw prediction interval band
    if prediction_interval is not None:
        if prediction_interval == "rmse":
            band_width = rmse
            band_label = f"±RMSE ({rmse:.2f})"
        elif prediction_interval == "95ci":
            band_width = 1.96 * rmse
            band_label = f"±1.96×RMSE (95% PI)"
        elif prediction_interval == "10pct":
            # 10% of mean observed value
            band_width = 0.10 * np.mean(y_true)
            band_label = "±10%"
        elif prediction_interval == "20pct":
            band_width = 0.20 * np.mean(y_true)
            band_label = "±20%"
        else:
            band_width = None
            band_label = None

        if band_width is not None:
            ax.fill_between(
                x_line,
                x_line - band_width,
                x_line + band_width,
                color="#3498db",
                alpha=0.15,
                label=band_label,
                zorder=1,
            )
            # Draw dashed lines at band edges
            ax.plot(x_line, x_line - band_width, "--", color="#3498db", alpha=0.4, linewidth=0.8, zorder=2)
            ax.plot(x_line, x_line + band_width, "--", color="#3498db", alpha=0.4, linewidth=0.8, zorder=2)

    # 1:1 line
    ax.plot(lims, lims, "k-", alpha=0.7, linewidth=1.5, label="1:1 line", zorder=3)

    # Scatter points
    ax.scatter(y_true, y_pred, alpha=0.6, s=30, edgecolors="w", linewidth=0.3, zorder=4)

    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("Observed Protein (%)")
    ax.set_ylabel("Predicted Protein (%)")
    title = f"Predicted vs Observed"
    if model_name:
        title += f" - {model_name}"
    ax.set_title(title)

    # Add metrics text
    if metrics:
        text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        ax.text(
            0.05, 0.95, text, transform=ax.transAxes,
            fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Add legend for prediction interval
    if prediction_interval is not None:
        ax.legend(loc="lower right", fontsize=7)

    ax.set_aspect("equal")
    return ax


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    title: str = "Feature Importance",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot top N feature importances as horizontal bar chart."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.3)))

    # Sort by importance
    idx = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    values = importances[idx]

    ax.barh(range(len(names)), values[::-1], align="center")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(title)

    return ax


def plot_shap_summary(
    model,
    X: np.ndarray,
    feature_names: List[str],
    max_display: int = 20,
) -> None:
    """Plot SHAP beeswarm summary."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(
            shap_values, X,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )
    except Exception as e:
        logger.warning("SHAP plot failed: %s", e)


def results_summary_table(
    results: Dict[str, Dict],
) -> pd.DataFrame:
    """Create summary table with mean ± std for each metric/model."""
    rows = []
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        for metric_name, values in metrics.items():
            if isinstance(values, list) and len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1) if len(values) > 1 else 0
                row[f"{metric_name}_mean"] = round(mean_val, 4)
                row[f"{metric_name}_std"] = round(std_val, 4)
                row[metric_name] = f"{mean_val:.4f} ± {std_val:.4f}"
        rows.append(row)

    return pd.DataFrame(rows)
