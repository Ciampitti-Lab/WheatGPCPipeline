"""Double logistic sigmoid fitting for GCVI curve modeling using lmfit."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("gpc_pipeline")


def double_logistic(
    t: np.ndarray,
    m1: float,
    m2: float,
    m3: float,
    m4: float,
    m5: float,
    m6: float,
    m7: float,
) -> np.ndarray:
    """Double logistic sigmoid: GCVI(t) = m1 + (m2 - m7*t) * (sig1 - sig2)."""
    sig1 = 1.0 / (1.0 + np.exp((m3 - t) / (m4 + 1e-8)))
    sig2 = 1.0 / (1.0 + np.exp((m5 - t) / (m6 + 1e-8)))
    return m1 + (m2 - m7 * t) * (sig1 - sig2)


def fit_double_logistic(
    t: np.ndarray,
    y: np.ndarray,
    max_nfev: int = 2000,
) -> Optional[Dict[str, float]]:
    """Fit double logistic to (t, y) using lmfit. Returns params dict or None."""
    try:
        from lmfit import Parameters, minimize

        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)

        # Remove NaN
        mask = np.isfinite(y) & np.isfinite(t)
        t_clean = t[mask]
        y_clean = y[mask]

        if len(t_clean) < 8:
            return None

        y_min, y_max = y_clean.min(), y_clean.max()
        t_min, t_max = t_clean.min(), t_clean.max()

        # Find approximate peak location from data for better initialization
        peak_idx = np.argmax(y_clean)
        t_peak = t_clean[peak_idx]

        # Estimate greenup (m3): where curve reaches ~20% of max (before peak)
        # Estimate senescence (m5): where curve drops to ~80% of max (after peak)
        threshold_low = y_min + 0.2 * (y_max - y_min)
        threshold_high = y_min + 0.8 * (y_max - y_min)

        # Find greenup point (before peak)
        before_peak = t_clean < t_peak
        if before_peak.any():
            below_thresh = before_peak & (y_clean < threshold_high)
            if below_thresh.any():
                m3_init = t_clean[below_thresh][-1]  # Last point below threshold before peak
            else:
                m3_init = t_peak - (t_peak - t_min) * 0.3
        else:
            m3_init = t_min + (t_peak - t_min) * 0.5

        # Find senescence point (after peak)
        after_peak = t_clean > t_peak
        if after_peak.any():
            below_thresh = after_peak & (y_clean < threshold_high)
            if below_thresh.any():
                m5_init = t_clean[after_peak][np.where(y_clean[after_peak] < threshold_high)[0][0]] if (y_clean[after_peak] < threshold_high).any() else t_peak + (t_max - t_peak) * 0.3
            else:
                m5_init = t_peak + (t_max - t_peak) * 0.3
        else:
            m5_init = t_peak + (t_max - t_peak) * 0.5

        # Ensure m3 < m5 with minimum gap
        min_gap = (t_max - t_min) * 0.1
        if m5_init <= m3_init + min_gap:
            m3_init = t_peak - min_gap
            m5_init = t_peak + min_gap

        params = Parameters()
        params.add("m1", value=y_min, min=0, max=y_max)
        params.add("m2", value=y_max - y_min, min=0.1, max=3 * (y_max - y_min + 0.01))
        params.add(
            "m3",
            value=m3_init,
            min=t_min,
            max=t_peak + min_gap,  # m3 must be before or at peak
        )
        params.add(
            "m4",
            value=(t_peak - t_min) * 0.15,
            min=1,
            max=(t_max - t_min) * 0.3,
        )
        params.add(
            "m5",
            value=m5_init,
            min=t_peak - min_gap,  # m5 must be at or after peak
            max=t_max,
        )
        params.add(
            "m6",
            value=(t_max - t_peak) * 0.15,
            min=1,
            max=(t_max - t_min) * 0.3,
        )
        params.add("m7", value=0, min=-0.05, max=0.05)

        def residual(p):
            vals = p.valuesdict()
            pred = double_logistic(
                t_clean,
                vals["m1"], vals["m2"], vals["m3"],
                vals["m4"], vals["m5"], vals["m6"], vals["m7"],
            )
            return y_clean - pred

        result = minimize(residual, params, method="leastsq", max_nfev=max_nfev)

        if result.success or result.nfev > 0:
            fitted_params = result.params.valuesdict()
            # Compute R² of fit
            y_pred = double_logistic(
                t_clean,
                fitted_params["m1"], fitted_params["m2"], fitted_params["m3"],
                fitted_params["m4"], fitted_params["m5"], fitted_params["m6"],
                fitted_params["m7"],
            )
            ss_res = np.sum((y_clean - y_pred) ** 2)
            ss_tot = np.sum((y_clean - y_clean.mean()) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))
            fitted_params["fit_r2"] = float(r2)
            return fitted_params
        return None

    except Exception as e:
        logger.debug("Double logistic fit failed: %s", e)
        return None


def find_peak_from_params(
    params: Dict[str, float],
    t_range: Tuple[float, float],
    n_points: int = 1000,
) -> Tuple[float, float]:
    """Find (peak_day, peak_value) from fitted double logistic params."""
    t = np.linspace(t_range[0], t_range[1], n_points)
    y = double_logistic(
        t,
        params["m1"], params["m2"], params["m3"],
        params["m4"], params["m5"], params["m6"], params["m7"],
    )
    idx = np.argmax(y)
    return float(t[idx]), float(y[idx])


def find_peak_fallback(t: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fallback peak detection using 5-point rolling mean."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    mask = np.isfinite(y) & np.isfinite(t)

    if mask.sum() < 3:
        mid = np.nanmedian(t)
        return float(mid), float(np.nanmax(y))

    t_clean = t[mask]
    y_clean = y[mask]

    if len(y_clean) >= 5:
        kernel = np.ones(5) / 5
        y_smooth = np.convolve(y_clean, kernel, mode="same")
    else:
        y_smooth = y_clean

    idx = np.argmax(y_smooth)
    return float(t_clean[idx]), float(y_smooth[idx])
