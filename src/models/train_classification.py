"""DVC stage — classification: predict churn / inactivity risk.

Target is the engineered ``churn_risk`` proxy (no transaction in the most recent
month). **Recall is the priority metric** — the bank would rather flag a
borderline-active customer than miss someone about to lapse.

Leakage guard: ``churn_risk`` is *defined* from ``last_txn_date`` /
``recency_days`` (recency beyond the window ⇒ churn), so both are excluded from
the feature set. ``gender``/``race`` are excluded for fairness. Models use
balanced class weights to counter the 18% positive rate.

Run via the pipeline (`dvc repro train_classification`) or directly:
    python -m src.models.train_classification
"""
from __future__ import annotations

import json

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config, tracking


def _preprocess(num_cols, cat_cols) -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("ohe", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, num_cols),
                              ("cat", categorical, cat_cols)])


def _models(rs):
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=rs,
                                                 class_weight="balanced", n_jobs=-1),
    }


def train(features_path=None) -> dict:
    params = config.load_params()
    clf = params["classification"]
    rs = clf["random_state"]
    features_path = features_path or config.CUSTOMER_FEATURES

    df = pd.read_parquet(features_path)
    cat_cols = clf["categorical"]
    num_cols = [c for c in clf["features"] if c not in cat_cols]
    X = df[num_cols + cat_cols]
    y = df[clf["target"]].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=clf["test_size"],
                                              random_state=rs, stratify=y)

    results, best = {}, {"name": None, "recall": -np.inf, "pipe": None, "report": None, "cm": None}
    with tracking.start_run("classification", run_name="churn-model-search"):
        mlflow.log_params({"n_features": len(X.columns), "n_train": len(X_tr),
                           "n_test": len(X_te), "positive_rate": float(y.mean())})
        for name, model in _models(rs).items():
            pipe = Pipeline([("prep", _preprocess(num_cols, cat_cols)), ("model", model)])
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_te)
            proba = pipe.predict_proba(X_te)[:, 1]
            m = {"recall": float(recall_score(y_te, pred)),
                 "precision": float(precision_score(y_te, pred, zero_division=0)),
                 "f1": float(f1_score(y_te, pred)),
                 "roc_auc": float(roc_auc_score(y_te, proba))}
            results[name] = m
            mlflow.log_metrics({f"{name}_{k}": v for k, v in m.items()})
            # Pick on recall (the business priority), tie-break implicitly by order.
            if m["recall"] > best["recall"]:
                best.update(name=name, recall=m["recall"], pipe=pipe,
                            report=classification_report(y_te, pred, output_dict=True),
                            cm=confusion_matrix(y_te, pred).tolist())

        mlflow.log_param("best_model", best["name"])
        mlflow.log_metric("best_recall", best["recall"])
        config.ensure_dirs()
        model_path = config.MODELS_DIR / "classification_churn.joblib"
        joblib.dump(best["pipe"], model_path)
        mlflow.sklearn.log_model(best["pipe"], "model")

    # Test-set probabilities for the report (ROC / threshold analysis).
    preds = pd.DataFrame({"churn_actual": y_te.to_numpy(),
                          "churn_proba": best["pipe"].predict_proba(X_te)[:, 1]})
    preds.to_parquet(config.PROCESSED_DIR / "churn_predictions.parquet", index=False)

    metrics = {"best_model": best["name"], "positive_rate": float(y.mean()),
               "n_train": len(X_tr), "n_test": len(X_te), "by_model": results,
               "confusion_matrix": best["cm"]}
    with open(config.METRICS_DIR / "classification.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"classification: best={best['name']} recall={best['recall']:.3f} "
          f"roc_auc={results[best['name']]['roc_auc']:.3f}")
    return metrics


if __name__ == "__main__":
    train()
