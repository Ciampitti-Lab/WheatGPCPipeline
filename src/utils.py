"""Utility functions for the GPC Pipeline."""

import json
import logging
import os
from pathlib import Path


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the pipeline logger."""
    logger = logging.getLogger("gpc_pipeline")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist, return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(data: dict, path: str) -> None:
    """Save checkpoint data as JSON for resume support."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_checkpoint(path: str) -> dict:
    """Load checkpoint data from JSON. Returns empty dict if not found."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def get_project_root() -> Path:
    """Return the project root directory (parent of src/)."""
    return Path(__file__).resolve().parent.parent
