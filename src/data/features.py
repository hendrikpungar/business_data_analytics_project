"""DVC stage 2 — build a customer-level feature table.

The raw data is transaction-level; segmentation, regression and
classification all operate per customer, so this stage aggregates
transactions up to one row per ``customer_id`` and engineers the
business features described in docs/project_description.txt.

Run via the pipeline (`dvc repro features`) or directly:
    python -m src.data.features
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise ratio that yields NaN (not inf) when the denominator is 0."""
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def build_features(cleaned_path=None, out_path=None) -> pd.DataFrame:
    params = config.load_params()
    cid = params["data"]["customer_id"]
    amount = params["data"]["amount_col"]
    cleaned_path = cleaned_path or config.CLEANED_DATA
    out_path = out_path or config.CUSTOMER_FEATURES

    df = pd.read_parquet(cleaned_path)

    # "As of" date = latest transaction observed; used for recency / account age.
    as_of = df["txn_date"].max()
    window_months = params["features"]["recent_activity_months"]
    cutoff = as_of - pd.DateOffset(months=window_months)

    grouped = df.groupby(cid)

    # Static customer attributes (constant per customer; take the first value).
    static_cols = [
        "credit_limit", "month_end_bal", "monthly_payment", "age",
        "total_cards", "card_type", "relationship", "gender",
        "customer_postcode", "date_opened",
    ]
    static = grouped[[c for c in static_cols if c in df.columns]].first()

    # Behavioural aggregates from the transactions.
    feats = pd.DataFrame({
        "total_spend": grouped[amount].sum(),
        "txn_count": grouped[amount].count(),
        "avg_txn_amount": grouped[amount].mean(),
        "max_txn_amount": grouped[amount].max(),
        "n_merchant_groups": grouped["merchant_group"].nunique(),
        "n_countries": grouped["txn_country"].nunique(),
        "last_txn_date": grouped["txn_date"].max(),
    })

    out = static.join(feats)

    # Engineered business ratios / derived features.
    out["credit_utilisation"] = _safe_ratio(out["month_end_bal"], out["credit_limit"])
    out["payment_ratio"] = _safe_ratio(out["monthly_payment"], out["month_end_bal"])
    out["account_age_days"] = (as_of - out["date_opened"]).dt.days
    out["recency_days"] = (as_of - out["last_txn_date"]).dt.days

    # Churn proxy: no transaction in the most recent `window_months` months.
    out["churn_risk"] = (out["last_txn_date"] < cutoff).astype(int)

    out = out.reset_index()
    config.ensure_dirs()
    out.to_parquet(out_path, index=False)
    print(
        f"features: {len(out):,} customers x {out.shape[1]} cols -> {out_path}\n"
        f"  as_of={as_of.date()} churn_cutoff={cutoff.date()} "
        f"churn_rate={out['churn_risk'].mean():.1%}"
    )
    return out


if __name__ == "__main__":
    build_features()
