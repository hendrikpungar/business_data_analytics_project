# Business Data Analytics Project — Credit Card Customers

Business analytics project on a credit card customer transaction dataset of approximately
100,000 records across about 1,000 clients. The goal is to transform raw financial and
transactional data into useful business insights that support decision-making in a
banking or financial services context — understanding customers, identifying business
opportunities, detecting risk, and making practical recommendations for customer
management, marketing, and profitability.

## Objectives

The project addresses four business analytics tracks:

1. **Customer Segmentation** — group customers into meaningful business segments
   (e.g. high-value, low-activity, revolvers, transactors) using K-Means on financial
   and behavioural features.
2. **Regression — CLTV prediction** — predict Customer Lifetime Value with Random
   Forest / Gradient Boosting regressors.
3. **Classification — Churn / Inactivity risk** — predict at-risk customers using a
   business-defined churn proxy (e.g. no transactions in the most recent month), with
   Logistic Regression and Random Forest.
4. **Forecasting — Monthly Spending** — short-term forecast of total monthly spending
   (and optionally by merchant group) using moving average, ARIMA, or lag-based
   regression.

## Repository structure

```
.
├── data/                  # data.csv lives here locally but is git-ignored
├── docs/                  # project description (txt) and reference material (pdf)
├── scripts/               # reference cleaning script from the instructor
├── notebooks/             # analysis notebooks (added in later steps)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Requires Python 3.11.

```powershell
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

To use the venv as a Jupyter kernel:

```powershell
python -m ipykernel install --user --name bda-project --display-name "Python 3.11 (bda-project)"
```

## Data

The raw dataset (`data/data.csv`, latin-1 encoded) is **not** committed — place it at
`data/data.csv` locally. Load with:

```python
import pandas as pd
df = pd.read_csv("data/data.csv", encoding="latin1")
```

Dates use a Julian `YYYYDDD` format (e.g. `2012092` → `2012-04-01`); see
`scripts/code_to_clean_data.txt` for the conversion helper.

## Deliverables

- Jupyter notebook with the full analysis
- PDF report with business findings and recommendations

The dataset itself is **not** submitted.
