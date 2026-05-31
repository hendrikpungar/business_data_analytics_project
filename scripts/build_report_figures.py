"""Regenerate the figures used in docs/report.md from the DVC pipeline outputs.

Reproducible report assets: run after `dvc repro`.
    python scripts/build_report_figures.py
Writes PNGs to docs/figures/ and prints the exact statistics cited in the
report so every claim stays verifiable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path
from src import config
from src.models.train_segmentation import _preprocess

sns.set_theme(style="whitegrid")
FIG = config.ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

txn = pd.read_parquet(config.CLEANED_DATA)
cust = pd.read_parquet(config.CUSTOMER_FEATURES)
seg = pd.read_parquet(config.PROCESSED_DIR / "customer_segments.parquet")
metrics = json.load(open(config.METRICS_DIR / "segmentation.json"))
seg_params = config.load_params()["segmentation"]

NAMES = {4: "High-Value Whales", 5: "Affluent Transactors", 3: "Engaged Revolvers",
         0: "Core Transactors", 1: "Credit-Constrained Revolvers", 2: "Dormant / At-Risk"}
assert seg.groupby("segment")["total_spend"].median().idxmax() == 4
seg["segment_name"] = seg["segment"].map(NAMES)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


# ---- EDA 1: merchant value vs volume -------------------------------------
mg = (txn.dropna(subset=["merchant_group"]).groupby("merchant_group")["amount"]
      .agg(spend="sum", volume="count"))
mg["avg"] = mg["spend"] / mg["volume"]
top = mg.sort_values("spend", ascending=False).head(12)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(top["volume"], top["spend"] / 1e6, s=60, color="#08519c")
for name, r in top.iterrows():
    ax.annotate(name, (r["volume"], r["spend"] / 1e6), fontsize=8,
                xytext=(4, 4), textcoords="offset points")
ax.set(xlabel="transactions (volume)", ylabel="total spend (EUR M, value)",
       title="Insurance drives spend value; Petrol drives transaction volume")
save(fig, "eda_merchant_value_volume.png")
print("Insurance spend EUR %.2fM (%.1f%% of spend); Petrol %d txns, avg EUR %.0f" % (
    mg.loc["Insurance", "spend"] / 1e6, 100 * mg.loc["Insurance", "spend"] / txn["amount"].sum(),
    mg.loc["Petrol", "volume"], mg.loc["Petrol", "avg"]))

# ---- EDA 2: monthly spend trend ------------------------------------------
m = (txn.set_index("txn_date").resample("MS")["amount"].agg(spend="sum", n="count"))
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(m.index[:-1], m["spend"].iloc[:-1] / 1e6, marker="o", color="#08519c", label="full months")
ax.plot(m.index[-1], m["spend"].iloc[-1] / 1e6, marker="o", ms=9, color="crimson")
ax.annotate("2014-03 partial\n(ends 28th)", (m.index[-1], m["spend"].iloc[-1] / 1e6),
            color="crimson", fontsize=8, xytext=(-10, 12), textcoords="offset points")
ax.set(xlabel="month", ylabel="total spend (EUR M)",
       title="Monthly card spend rose 29% from 2013-10 to the 2014-01 peak")
save(fig, "eda_monthly_trend.png")
g = m["spend"].loc["2013-10-01"], m["spend"].loc["2014-01-01"]
print("monthly spend 2013-10 EUR %.2fM -> 2014-01 EUR %.2fM (+%.0f%%); last month n=%d vs median n=%d" % (
    g[0] / 1e6, g[1] / 1e6, 100 * (g[1] / g[0] - 1), m["n"].iloc[-1], int(m["n"].iloc[:-1].median())))

# ---- EDA 3: credit utilisation -------------------------------------------
u = cust["credit_utilisation"].clip(-0.5, 2).dropna()
share = (cust["credit_utilisation"] > 0.8).mean()
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(u, bins=40, color="#9ecae1", edgecolor="white")
ax.axvline(0.8, color="crimson", ls="--")
ax.annotate(f"{share:.0%} above 0.8\n(high-risk band)", (0.82, ax.get_ylim()[1] * 0.7),
            color="crimson", fontsize=9)
ax.set(xlabel="credit utilisation (month-end balance / credit limit)", ylabel="customers",
       title=f"{share:.0%} of customers sit in the >0.8 high-utilisation risk band")
save(fig, "eda_utilisation.png")
print("utilisation >0.8: %.1f%%; mean %.2f median %.2f" % (
    100 * share, cust["credit_utilisation"].mean(), cust["credit_utilisation"].median()))

# ---- SEG 1: silhouette by k ----------------------------------------------
sk = {int(k): v for k, v in metrics["silhouette_by_k"].items()}
best = metrics["best_k"]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(list(sk), list(sk.values()), marker="o", color="#08519c")
ax.axvline(best, color="crimson", ls="--")
ax.annotate(f"peak k={best}\n{sk[best]:.3f}", (best, sk[best]), color="crimson",
            fontsize=9, xytext=(10, -4), textcoords="offset points")
ax.set(xlabel="k (clusters)", ylabel="silhouette",
       title="K-Means silhouette peaks at k=6 (0.41) — a genuine elbow")
save(fig, "seg_silhouette.png")

# ---- profiling tables -----------------------------------------------------
n = len(seg)
prof = seg.groupby(["segment", "segment_name"]).agg(
    customers=("segment", "size"),
    total_spend=("total_spend", "median"), txn_count=("txn_count", "median"),
    avg_txn_amount=("avg_txn_amount", "median"), credit_limit=("credit_limit", "median"),
    credit_utilisation=("credit_utilisation", "median"), payment_ratio=("payment_ratio", "median"),
    n_merchant_groups=("n_merchant_groups", "median"), recency_days=("recency_days", "median"),
    churn_risk=("churn_risk", "mean"),
)
prof["pct_customers"] = prof["customers"] / n * 100
prof["pct_of_spend"] = seg.groupby(["segment", "segment_name"])["total_spend"].sum() / seg["total_spend"].sum() * 100
prof = prof.sort_values("pct_of_spend", ascending=False)
order = prof.index.get_level_values("segment_name")

# ---- SEG 2: economic weight ----------------------------------------------
x = np.arange(len(order)); w = 0.38
fig, ax = plt.subplots(figsize=(9.5, 4.8))
ax.bar(x - w / 2, prof["pct_customers"], w, label="% of customers", color="#9ecae1")
ax.bar(x + w / 2, prof["pct_of_spend"], w, label="% of spend", color="#08519c")
for i, (c, s) in enumerate(zip(prof["pct_customers"], prof["pct_of_spend"])):
    ax.text(i - w / 2, c + 0.4, f"{c:.0f}", ha="center", fontsize=8)
    ax.text(i + w / 2, s + 0.4, f"{s:.0f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(order, rotation=25, ha="right")
ax.set(ylabel="share (%)",
       title="23% of customers (Whales + Affluent Transactors) generate 71% of spend")
ax.legend()
save(fig, "seg_economic_weight.png")

# ---- SEG 3: profile heatmap ----------------------------------------------
PCOLS = ["total_spend", "txn_count", "avg_txn_amount", "credit_limit",
         "credit_utilisation", "payment_ratio", "n_merchant_groups", "recency_days", "churn_risk"]
hm = seg.groupby("segment_name")[PCOLS].median().loc[order]
hmz = (hm - hm.mean()) / hm.std()
fig, ax = plt.subplots(figsize=(10, 4.6))
sns.heatmap(hmz, annot=hm.round(2), fmt=".2f", cmap="RdBu_r", center=0,
            cbar_kws={"label": "z-score across segments"}, ax=ax)
ax.set(title="Segment profiles — colour = relative level, text = median (original units)", ylabel="")
save(fig, "seg_profile_heatmap.png")

# ---- SEG 4: PCA map -------------------------------------------------------
Xp = _preprocess(seg_params).fit_transform(seg[seg_params["features"]])
fig, ax = plt.subplots(figsize=(7.5, 6))
for name in order:
    msk = (seg["segment_name"] == name).to_numpy()
    ax.scatter(Xp[msk, 0], Xp[msk, 1], s=12, alpha=0.6, label=name)
ax.set(xlabel="PCA 1", ylabel="PCA 2", title="Six customer segments separate cleanly in PCA space")
ax.legend(fontsize=8, markerscale=2)
save(fig, "seg_pca_map.png")

print("\n=== SEGMENT PROFILE (cite in report) ===")
pd.set_option("display.width", 250, "display.max_columns", 30)
print(prof.round(2).to_string())
print("\ntop-2 segments: customers=%d (%.1f%%), spend share=%.1f%%" % (
    prof["customers"].iloc[:2].sum(), prof["pct_customers"].iloc[:2].sum(), prof["pct_of_spend"].iloc[:2].sum()))
print("bottom-2 segments: customers=%d (%.1f%%), spend share=%.1f%%" % (
    prof["customers"].iloc[-2:].sum(), prof["pct_customers"].iloc[-2:].sum(), prof["pct_of_spend"].iloc[-2:].sum()))


# ======================= REGRESSION (§5) ===================================
import joblib
from sklearn.metrics import auc, roc_curve

reg_m = json.load(open(config.METRICS_DIR / "regression.json"))
reg_pred = pd.read_parquet(config.PROCESSED_DIR / "cltv_predictions.parquet")
reg_model = joblib.load(config.MODELS_DIR / "regression_cltv.joblib")

fig, ax = plt.subplots(figsize=(6.2, 6))
lim = [0, reg_pred["cltv_actual"].quantile(0.99)]
ax.scatter(reg_pred["cltv_actual"], reg_pred["cltv_pred"], s=14, alpha=0.5, color="#08519c")
ax.plot(lim, lim, "--", color="crimson", label="perfect prediction")
r2 = reg_m["by_model"][reg_m["best_model"]]["r2"]
ax.set(xlim=lim, ylim=lim, xlabel="actual CLTV (EUR)", ylabel="predicted CLTV (EUR)",
       title=f"CLTV model explains {r2:.0%} of value variance (R²={r2:.2f})")
ax.legend()
save(fig, "reg_pred_vs_actual.png")

names = reg_model.named_steps["prep"].get_feature_names_out()
imp = pd.Series(reg_model.named_steps["model"].feature_importances_, index=names).sort_values().tail(10)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.barh(imp.index, imp.values, color="#08519c")
ax.set(xlabel="feature importance", title="Monthly payment, merchant breadth and balance drive predicted CLTV")
save(fig, "reg_feature_importance.png")

# ======================= CLASSIFICATION (§6) ===============================
clf_m = json.load(open(config.METRICS_DIR / "classification.json"))
clf_pred = pd.read_parquet(config.PROCESSED_DIR / "churn_predictions.parquet")

cm = np.array(clf_m["confusion_matrix"])
fig, ax = plt.subplots(figsize=(5, 4.2))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["pred stay", "pred churn"], yticklabels=["actual stay", "actual churn"], ax=ax)
ax.set(title=f"Churn model catches {cm[1,1]} of {cm[1].sum()} at-risk customers (89% recall)")
save(fig, "clf_confusion.png")

fpr, tpr, _ = roc_curve(clf_pred["churn_actual"], clf_pred["churn_proba"])
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#08519c", lw=2, label=f"Logistic Regression (AUC={auc(fpr,tpr):.2f})")
ax.plot([0, 1], [0, 1], "--", color="grey")
ax.set(xlabel="false positive rate", ylabel="recall (true positive rate)",
       title="Churn model ROC — AUC 0.79")
ax.legend()
save(fig, "clf_roc.png")

# ======================= FORECASTING (§7) ==================================
fc_m = json.load(open(config.METRICS_DIR / "forecasting.json"))
fc = pd.read_parquet(config.PROCESSED_DIR / "spend_forecast.parquet")
fc["month"] = pd.to_datetime(fc["month"])
hist, fut = fc[fc["kind"] == "actual"], fc[fc["kind"] == "forecast"]
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(hist["month"], hist["spend"] / 1e6, marker="o", color="#08519c", label="actual")
bridge = pd.concat([hist.iloc[[-1]], fut])
ax.plot(bridge["month"], bridge["spend"] / 1e6, marker="o", ls="--", color="crimson", label="forecast")
ax.set(xlabel="month", ylabel="total spend (EUR M)",
       title=f"Portfolio spend forecast to hold near EUR{fut['spend'].mean()/1e6:.1f}M/month (MAPE {fc_m['by_method'][fc_m['best_method']]['mape']:.0f}%)")
ax.legend()
save(fig, "fc_forecast.png")
print("\nregression best %s R2=%.3f | churn recall %.3f | forecast %.2fM/mo" % (
    reg_m["best_model"], r2, clf_m["by_model"][clf_m["best_model"]]["recall"], fut["spend"].mean() / 1e6))
