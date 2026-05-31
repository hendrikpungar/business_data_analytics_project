"""DVC stage 3 — customer segmentation (K-Means), tracked with MLflow.

This is the reference template for the four modelling tracks: load the
customer feature table, build a leakage-free preprocessing pipeline,
search over k, log everything to MLflow, and emit DVC-tracked metric and
output files.

Run via the pipeline (`dvc repro train_segmentation`) or directly:
    python -m src.models.train_segmentation
"""
from __future__ import annotations

import json

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

from src import config, tracking


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to quantiles learned on the training data.

    The customer features are heavily right-skewed (total_spend skew ~8) and a
    few derived ratios take pathological values (payment_ratio down to -23,715
    from negative balances). Capping extremes stops these whales/outliers from
    dominating the Euclidean distance in PCA space and collapsing K-Means to
    k=2. Quantiles are fit (not recomputed at transform time) so the step stays
    leakage-free if reused on held-out data.
    """

    def __init__(self, q_low: float = 0.01, q_high: float = 0.99):
        self.q_low = q_low
        self.q_high = q_high

    def fit(self, X, y=None):
        Xa = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(Xa, self.q_low, axis=0)
        self.upper_ = np.nanquantile(Xa, self.q_high, axis=0)
        return self

    def transform(self, X):
        Xa = np.asarray(X, dtype=float)
        return np.clip(Xa, self.lower_, self.upper_)


def _preprocess(seg_params) -> Pipeline:
    # Winsorise -> median-impute -> Yeo-Johnson de-skew (also standardises,
    # handles negatives) -> PCA. Order matters for distance-based clustering:
    # tame outliers and skew before measuring distances.
    q_low, q_high = seg_params["winsor_quantiles"]
    return Pipeline([
        ("winsor", Winsorizer(q_low=q_low, q_high=q_high)),
        ("imputer", SimpleImputer(strategy="median")),
        ("power", PowerTransformer(method="yeo-johnson", standardize=True)),
        ("pca", PCA(n_components=seg_params["pca_components"],
                    random_state=seg_params["random_state"])),
    ])


def train(features_path=None) -> dict:
    params = config.load_params()
    seg = params["segmentation"]
    features_path = features_path or config.CUSTOMER_FEATURES

    df = pd.read_parquet(features_path)
    X = df[seg["features"]]

    prep = _preprocess(seg)
    X_prep = prep.fit_transform(X)

    # Search k, pick the best silhouette (the report should still sanity-check
    # the winner against business meaning, not silhouette alone).
    results = {}
    best = {"k": None, "silhouette": -1.0, "model": None, "labels": None}
    with tracking.start_run("segmentation", run_name="kmeans-search"):
        mlflow.log_params({f"feature_{i}": f for i, f in enumerate(seg["features"])})
        mlflow.log_param("pca_components", seg["pca_components"])
        for k in range(seg["k_min"], seg["k_max"] + 1):
            km = KMeans(n_clusters=k, n_init=20, random_state=seg["random_state"])
            labels = km.fit_predict(X_prep)
            score = float(silhouette_score(X_prep, labels))
            results[k] = score
            mlflow.log_metric("silhouette", score, step=k)
            if score > best["silhouette"]:
                best.update(k=k, silhouette=score, model=km, labels=labels)

        mlflow.log_metric("best_k", best["k"])
        mlflow.log_metric("best_silhouette", best["silhouette"])

        # Persist the fitted preprocessing + clusterer together.
        config.ensure_dirs()
        full = Pipeline([("prep", prep), ("kmeans", best["model"])])
        model_path = config.MODELS_DIR / "segmentation_kmeans.joblib"
        joblib.dump(full, model_path)
        mlflow.sklearn.log_model(full, "model")

    # Cluster assignments for downstream profiling / the report.
    df_out = df.copy()
    df_out["segment"] = best["labels"]
    df_out.to_parquet(config.PROCESSED_DIR / "customer_segments.parquet", index=False)

    # DVC-tracked metrics file.
    metrics = {
        "best_k": best["k"],
        "best_silhouette": best["silhouette"],
        "silhouette_by_k": results,
    }
    with open(config.METRICS_DIR / "segmentation.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"segmentation: best k={best['k']} silhouette={best['silhouette']:.3f}")
    return metrics


if __name__ == "__main__":
    train()
