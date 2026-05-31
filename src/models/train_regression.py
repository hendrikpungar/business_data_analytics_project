"""DVC stage — regression: predict Customer Lifetime Value (CLTV).

CLTV is a *derived* target: ``cltv = total_spend * revenue_rate`` (the revenue
rate is an explicit business assumption in params.yaml). Because the target is a
linear function of ``total_spend``, the spend-arithmetic features
(total_spend, txn_count, avg/max txn amount) are **excluded to avoid leakage** —
the model predicts value from *account and engagement* attributes a bank knows
relatively independently of the full transaction ledger. ``gender``/``race`` are
excluded for fairness.

Mirrors the segmentation template: ColumnTransformer preprocessing, several
models compared, everything logged to MLflow, DVC-tracked metric + model files.

Run via the pipeline (`dvc repro train_regression`) or directly:
    python -m src.models.train_regression
"""
from __future__ import annotations

import json

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
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
        "linear": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=rs, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=rs),
    }


def train(features_path=None) -> dict:
    params = config.load_params()
    reg = params["regression"]
    rs = reg["random_state"]
    features_path = features_path or config.CUSTOMER_FEATURES

    df = pd.read_parquet(features_path)
    df["cltv"] = df["total_spend"] * reg["revenue_rate"]

    cat_cols = reg["categorical"]
    num_cols = [c for c in reg["features"] if c not in cat_cols]
    X = df[num_cols + cat_cols]
    y = df["cltv"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=reg["test_size"], random_state=rs)

    results, best = {}, {"name": None, "r2": -np.inf, "pipe": None}
    with tracking.start_run("regression", run_name="cltv-model-search"):
        mlflow.log_params({"revenue_rate": reg["revenue_rate"], "n_features": len(X.columns),
                           "n_train": len(X_tr), "n_test": len(X_te)})
        for name, model in _models(rs).items():
            pipe = Pipeline([("prep", _preprocess(num_cols, cat_cols)), ("model", model)])
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_te)
            r2 = float(r2_score(y_te, pred))
            mae = float(mean_absolute_error(y_te, pred))
            rmse = float(np.sqrt(np.mean((y_te - pred) ** 2)))
            results[name] = {"r2": r2, "mae": mae, "rmse": rmse}
            mlflow.log_metrics({f"{name}_r2": r2, f"{name}_mae": mae, f"{name}_rmse": rmse})
            if r2 > best["r2"]:
                best.update(name=name, r2=r2, pipe=pipe)

        mlflow.log_param("best_model", best["name"])
        mlflow.log_metric("best_r2", best["r2"])
        config.ensure_dirs()
        model_path = config.MODELS_DIR / "regression_cltv.joblib"
        joblib.dump(best["pipe"], model_path)
        mlflow.sklearn.log_model(best["pipe"], "model")

    # Test-set predictions for the report (actual vs predicted).
    preds = pd.DataFrame({"cltv_actual": y_te, "cltv_pred": best["pipe"].predict(X_te)})
    preds.to_parquet(config.PROCESSED_DIR / "cltv_predictions.parquet", index=False)

    metrics = {"best_model": best["name"], "revenue_rate": reg["revenue_rate"],
               "n_train": len(X_tr), "n_test": len(X_te), "by_model": results}
    with open(config.METRICS_DIR / "regression.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"regression: best={best['name']} R2={best['r2']:.3f} "
          f"MAE={results[best['name']]['mae']:,.0f}")
    return metrics


if __name__ == "__main__":
    train()
