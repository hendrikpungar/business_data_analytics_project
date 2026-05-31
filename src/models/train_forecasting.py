"""DVC stage — forecasting: short-term monthly total card spend.

Aggregates the transaction ledger to a monthly spend series, trims partial
months at the ends (the dataset starts 2012-04 and ends mid-month on
2014-03-28), holds out the last ``test_months`` for evaluation, and compares a
moving-average baseline against ARIMA. The lower-error method is refit on the
full series and used to forecast ``horizon`` months ahead.

Run via the pipeline (`dvc repro train_forecasting`) or directly:
    python -m src.models.train_forecasting
"""
from __future__ import annotations

import json
import warnings

import mlflow
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src import config, tracking

warnings.filterwarnings("ignore")  # statsmodels convergence chatter on short series


def monthly_series(df: pd.DataFrame, amount: str, min_share: float) -> pd.Series:
    """Monthly total spend with partial end-months (low txn count) trimmed."""
    g = df.set_index("txn_date").resample("MS")[amount].agg(["sum", "count"])
    full = g["count"] >= min_share * g["count"].median()
    # Trim only leading/trailing partials, keep the interior intact.
    keep = full.cumsum().gt(0) & full[::-1].cumsum()[::-1].gt(0)
    return g.loc[keep, "sum"]


def _metrics(actual, pred) -> dict:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
    return {"rmse": rmse, "mape": mape}


def train(cleaned_path=None) -> dict:
    params = config.load_params()
    fc = params["forecasting"]
    amount = params["data"]["amount_col"]
    cleaned_path = cleaned_path or config.CLEANED_DATA

    df = pd.read_parquet(cleaned_path)
    series = monthly_series(df, amount, fc["min_month_share"])
    test_n, window, horizon = fc["test_months"], fc["ma_window"], fc["horizon"]

    train_s, test_s = series.iloc[:-test_n], series.iloc[-test_n:]

    # --- Method 1: moving-average baseline (flat last-window mean) -----------
    ma_value = train_s.iloc[-window:].mean()
    ma_pred = np.repeat(ma_value, test_n)
    ma_m = _metrics(test_s, ma_pred)

    # --- Method 2: ARIMA -----------------------------------------------------
    order = tuple(fc["arima_order"])
    arima_fit = ARIMA(train_s, order=order).fit()
    arima_pred = arima_fit.forecast(test_n).to_numpy()
    arima_m = _metrics(test_s, arima_pred)

    results = {"moving_average": ma_m, "arima": arima_m}
    best_name = min(results, key=lambda k: results[k]["rmse"])

    with tracking.start_run("forecasting", run_name="monthly-spend"):
        mlflow.log_params({"arima_order": str(order), "ma_window": window,
                           "test_months": test_n, "horizon": horizon,
                           "n_months": len(series)})
        for name, m in results.items():
            mlflow.log_metrics({f"{name}_rmse": m["rmse"], f"{name}_mape": m["mape"]})
        mlflow.log_param("best_method", best_name)

    # --- Refit the winner on the full series and forecast ahead --------------
    future_idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(),
                               periods=horizon, freq="MS")
    if best_name == "arima":
        fut = ARIMA(series, order=order).fit().forecast(horizon).to_numpy()
    else:
        fut = np.repeat(series.iloc[-window:].mean(), horizon)

    config.ensure_dirs()
    out = pd.concat([
        pd.DataFrame({"month": series.index, "spend": series.to_numpy(), "kind": "actual"}),
        pd.DataFrame({"month": future_idx, "spend": fut, "kind": "forecast"}),
    ], ignore_index=True)
    out.to_parquet(config.PROCESSED_DIR / "spend_forecast.parquet", index=False)

    metrics = {"best_method": best_name, "n_months": len(series),
               "train_months": len(train_s), "test_months": test_n,
               "horizon": horizon, "by_method": results,
               "forecast": {str(d.date()): float(v) for d, v in zip(future_idx, fut)}}
    with open(config.METRICS_DIR / "forecasting.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"forecasting: best={best_name} "
          f"MAPE={results[best_name]['mape']:.1f}% over {test_n}-month holdout; "
          f"{len(series)} full months")
    return metrics


if __name__ == "__main__":
    train()
