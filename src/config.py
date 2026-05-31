"""Project-wide paths and parameter loading.

Paths are derived from the repo root so scripts work regardless of the
working directory they are invoked from (DVC runs them from the root).
"""
from __future__ import annotations

from pathlib import Path

import yaml

# Repo root = two levels up from this file (src/config.py -> src -> root).
ROOT = Path(__file__).resolve().parents[1]

# Data layout (DVC-tracked). Raw data is git-ignored and tracked by DVC.
DATA_DIR = ROOT / "data"
RAW_DATA = DATA_DIR / "data.csv"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

CLEANED_DATA = INTERIM_DIR / "cleaned.parquet"
CUSTOMER_FEATURES = PROCESSED_DIR / "customer_features.parquet"

# Model / metric artifacts.
MODELS_DIR = ROOT / "models"
METRICS_DIR = ROOT / "metrics"

# MLflow: log to a local ./mlruns store by default. Override with the
# MLFLOW_TRACKING_URI environment variable (e.g. a remote tracking server).
MLFLOW_TRACKING_URI = f"file:///{(ROOT / 'mlruns').as_posix()}"
MLFLOW_EXPERIMENT = "bda-credit-card"

PARAMS_FILE = ROOT / "params.yaml"


def load_params() -> dict:
    """Load params.yaml (the single source of truth for stage parameters)."""
    with open(PARAMS_FILE, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_dirs() -> None:
    """Create output directories if they do not yet exist."""
    for d in (INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)
