"""Configuration management for the GPC Pipeline."""

from pathlib import Path
from typing import Any, Dict

import yaml

from .utils import get_project_root

# ---------------------------------------------------------------------------
# Feature category constants
# ---------------------------------------------------------------------------
VEGETATION_INDICES = [
    "NDVI", "EVI2", "GNDVI", "NDRE", "CIre", "IRECI", "NDWI", "MSI",
    "GCVI", "MNDWI", "NBR", "NBR2", "GVI", "DGCI", "NDSWIR1RedEdge1",
    "TCARI", "OSAVI", "CCCI", "mARI", "ExG", "DBSI", "SMMI", "MIRBI",
    "CVI", "S2REP", "PVI", "MTCI", "SAVI", "NDRE2", "RE_ratio", "PSRI",
    "LSWI",
]

SPECTRAL_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

METEO_VARS = ["t2m_mean", "t2m_min", "t2m_max", "precip", "pet", "dewpoint", "ssrd_MJm2"]

TEMPERATURE_VARS = ["t2m_mean", "t2m_min", "t2m_max"]
PRECIPITATION_VARS = ["precip"]
PET_VARS = ["pet"]
VPD_VARS = ["vpd"]
SSRD_VARS = ["ssrd_MJm2"]


def load_config(path: str | None = None) -> Dict[str, Any]:
    """Load YAML config. Defaults to configs/config.yaml."""
    if path is None:
        path = get_project_root() / "configs" / "config.yaml"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_feature_type(feature_name: str) -> str:
    """Return feature category for aggregation."""
    if feature_name in VEGETATION_INDICES:
        return "vegetation_indices"
    if feature_name in SPECTRAL_BANDS:
        return "spectral_bands"
    if feature_name in TEMPERATURE_VARS or feature_name == "dewpoint":
        return "temperature"
    if feature_name in PRECIPITATION_VARS:
        return "precipitation"
    if feature_name in PET_VARS:
        return "pet"
    if feature_name == "gdd":
        return "gdd"
    if feature_name in VPD_VARS or feature_name == "vpd":
        return "vpd"
    if feature_name in SSRD_VARS:
        return "ssrd"
    return "unknown"
