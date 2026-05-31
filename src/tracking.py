"""Thin MLflow helper so every modelling stage logs the same way.

Usage:

    from src import tracking

    with tracking.start_run("segmentation", run_name="kmeans-k4"):
        mlflow.log_params(...)
        mlflow.log_metric("silhouette", score)
        mlflow.sklearn.log_model(pipeline, "model")
"""
from __future__ import annotations

import contextlib

import mlflow

from src import config


def configure() -> None:
    """Point MLflow at the project's tracking store and experiment."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)


@contextlib.contextmanager
def start_run(track: str, run_name: str | None = None):
    """Open an MLflow run tagged with the analysis track.

    `track` is one of the four project tracks (segmentation, regression,
    classification, forecasting) and is stored as a tag so runs can be
    filtered per track in the MLflow UI.
    """
    configure()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("track", track)
        yield run

