"""Saxton & Rawls (2006) pedotransfer functions for soil water characteristics."""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger("gpc_pipeline")


def saxton_rawls_2006(
    sand_pct: float,
    clay_pct: float,
    om_pct: float,
) -> Dict[str, float]:
    """Saxton & Rawls (2006) pedotransfer: soil water characteristics from texture + OM."""
    # Convert to fractions
    pS = sand_pct / 100.0
    pC = clay_pct / 100.0
    pOM = om_pct / 100.0

    # Wilting Point (1500 kPa)
    theta_1500t = (
        -0.024 * pS
        + 0.487 * pC
        + 0.006 * pOM
        + 0.005 * pS * pOM
        - 0.013 * pC * pOM
        + 0.068 * pS * pC
        + 0.031
    )
    theta_1500 = theta_1500t + (0.14 * theta_1500t - 0.02)

    # Field Capacity (33 kPa)
    theta_33t = (
        -0.251 * pS
        + 0.195 * pC
        + 0.011 * pOM
        + 0.006 * pS * pOM
        - 0.027 * pC * pOM
        + 0.452 * pS * pC
        + 0.299
    )
    theta_33 = theta_33t + (1.283 * theta_33t**2 - 0.374 * theta_33t - 0.015)

    # Saturation (0 kPa)
    theta_S33t = (
        0.278 * pS
        + 0.034 * pC
        + 0.022 * pOM
        - 0.018 * pS * pOM
        - 0.027 * pC * pOM
        - 0.584 * pS * pC
        + 0.078
    )
    theta_S33 = theta_S33t + (0.636 * theta_S33t - 0.107)
    theta_SAT = theta_33 + theta_S33 - 0.097 * pS + 0.043

    # Bulk Density
    BD = (1 - theta_SAT) * 2.65

    # Saturated Hydraulic Conductivity
    lam = 1.0 / (np.log(33) - np.log(1500)) * (np.log(theta_33) - np.log(theta_1500))
    lam = max(lam, 0.01)

    B_val = 1.0 / lam
    KSAT = 1930 * (theta_SAT - theta_33) ** (3 - lam)

    # Air entry tension
    psi_e = (
        -21.67 * pS
        - 27.93 * pC
        - 81.97 * theta_S33
        + 71.12 * pS * theta_S33
        + 8.29 * pC * theta_S33
        + 14.05 * pS * pC
        + 27.16
    )

    # AWC
    AWC = theta_33 - theta_1500

    # SWCON
    SWCON = 0.15 + min(KSAT, 75) / 100.0

    # PAW
    PAW_mm_cm = AWC * 10.0

    return {
        "LL15": float(theta_1500),
        "DUL": float(theta_33),
        "SAT": float(theta_SAT),
        "BD": float(BD),
        "KSAT": float(KSAT),
        "AWC": float(AWC),
        "SWCON": float(SWCON),
        "lambda_val": float(lam),
        "B": float(B_val),
        "psi_e": float(psi_e),
        "PAW_mm_cm": float(PAW_mm_cm),
    }


def add_saxton_features(
    df: pd.DataFrame,
    sand_col: str = "soil_top_sand",
    clay_col: str = "soil_top_clay",
    om_col: str = "soil_top_om",
    prefix: str = "saxton_",
) -> pd.DataFrame:
    """Add Saxton-Rawls derived soil water columns to a DataFrame."""
    out = df.copy()

    sr_cols = {
        f"{prefix}LL15": [],
        f"{prefix}DUL": [],
        f"{prefix}SAT": [],
        f"{prefix}BD": [],
        f"{prefix}KSAT": [],
        f"{prefix}AWC": [],
        f"{prefix}SWCON": [],
        f"{prefix}lambda": [],
        f"{prefix}B": [],
        f"{prefix}psi_e": [],
        f"{prefix}PAW_mm_cm": [],
    }

    for _, row in df.iterrows():
        sand = row.get(sand_col, np.nan)
        clay = row.get(clay_col, np.nan)
        om = row.get(om_col, np.nan)

        if pd.isna(sand) or pd.isna(clay) or pd.isna(om):
            for col in sr_cols:
                sr_cols[col].append(np.nan)
            continue

        result = saxton_rawls_2006(sand, clay, om)
        sr_cols[f"{prefix}LL15"].append(result["LL15"])
        sr_cols[f"{prefix}DUL"].append(result["DUL"])
        sr_cols[f"{prefix}SAT"].append(result["SAT"])
        sr_cols[f"{prefix}BD"].append(result["BD"])
        sr_cols[f"{prefix}KSAT"].append(result["KSAT"])
        sr_cols[f"{prefix}AWC"].append(result["AWC"])
        sr_cols[f"{prefix}SWCON"].append(result["SWCON"])
        sr_cols[f"{prefix}lambda"].append(result["lambda_val"])
        sr_cols[f"{prefix}B"].append(result["B"])
        sr_cols[f"{prefix}psi_e"].append(result["psi_e"])
        sr_cols[f"{prefix}PAW_mm_cm"].append(result["PAW_mm_cm"])

    for col, values in sr_cols.items():
        out[col] = values

    n_valid = out[f"{prefix}AWC"].notna().sum()
    logger.info("Saxton-Rawls: %d/%d fields with valid soil data", n_valid, len(out))

    return out
