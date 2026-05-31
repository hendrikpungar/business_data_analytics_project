"""DVC stage 1 — clean the raw transaction CSV.

Reads the git-ignored, latin-1 encoded raw export and produces a typed
parquet file: lower-cased column names and Julian ``YYYYDDD`` date columns
converted to real datetimes. Mirrors scripts/code_to_clean_data.txt.

Run via the pipeline (`dvc repro clean`) or directly:
    python -m src.data.clean
"""
from __future__ import annotations

import pandas as pd

from src import config


def convert_julian_date(x) -> pd.Timestamp:
    """Convert a ``YYYYDDD`` Julian date (e.g. 2012092 -> 2012-04-01)."""
    if pd.isna(x):
        return pd.NaT
    s = str(int(float(x))).strip()
    if len(s) != 7:
        return pd.NaT
    return pd.to_datetime(s[:4] + s[4:], format="%Y%j", errors="coerce")


def clean(raw_path=None, out_path=None) -> pd.DataFrame:
    params = config.load_params()
    raw_path = raw_path or config.RAW_DATA
    out_path = out_path or config.CLEANED_DATA

    df = pd.read_csv(raw_path, encoding=params["data"]["encoding"], low_memory=False)
    df.columns = df.columns.str.lower().str.strip()

    for col in params["data"]["date_columns"]:
        if col in df.columns:
            df[col] = df[col].apply(convert_julian_date)

    # some age values are implausibly high. Let's drop them to avoid skewing age-based features.
    if params["data"].get("max_age", None) is not None and "age" in df.columns:
        before = len(df)
        df = df[df["age"] <= params["data"]["max_age"]].copy()
        n_dropped = before - len(df)
        print(f"dropped {n_dropped:,} rows with age > {params['data']['max_age']}")

    # Drop fully-duplicated rows: ~1,190 records are byte-identical across all
    # 22 columns (incl. posting_date), with multiplicities up to 14x. With no
    # genuine transaction key (txn_id is a 9-value code) these are export
    # artifacts, not real repeat purchases, and they inflate txn_count/frequency
    # features. See docs/project_description.txt; documented as a limitation.
    n_dropped = 0
    if params["data"].get("drop_duplicates", False):
        before = len(df)
        df = df.drop_duplicates(ignore_index=True)
        n_dropped = before - len(df)
       

    config.ensure_dirs()
    df.to_parquet(out_path, index=False)
    print(
        f"cleaned: {len(df):,} rows x {df.shape[1]} cols -> {out_path}"
        f" (dropped {n_dropped:,} duplicate rows)"
    )
    return df


if __name__ == "__main__":
    clean()
