# Payment Fraud Risk Scoring

End-to-end applied-ML project that scores card transactions by fraud risk on a
severely imbalanced dataset (~0.17% fraud). The deliverable is a single,
well-explained notebook plus a small reusable scoring layer.

## Problem

Card fraud is rare but expensive. The job of a fraud model is to **rank
transactions by risk** so a review team (or an auto-block rule) handles the
riskiest first. The hard part is the imbalance: only ~1 in 600 transactions is
fraud, so plain **accuracy is meaningless** — a model that flags nothing is
already 99.8% "accurate". We therefore optimise and report:

- **PR-AUC** (average precision) — primary model-selection metric; unlike
  ROC-AUC it isn't inflated by the huge true-negative count.
- **Recall** — share of actual fraud caught (the costly miss).
- **Precision** — share of flags that are truly fraud (analyst-effort cost).
- **F1 / F2**, **confusion matrix**, and **Precision@K / Recall@K** for a
  review-queue view.

Because precision and recall trade off, the decision **threshold is tuned
explicitly** (on validation) rather than left at 0.5.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(ULB Machine Learning Group), Kaggle slug `mlg-ulb/creditcardfraud`.

- 284,807 real European card transactions over two days in Sept 2013.
- 492 fraud (0.173%) in the raw file; **1,081 duplicate rows are dropped**,
  leaving 283,726 transactions / 473 fraud (0.167%).
- Features `V1`–`V28` are PCA components (anonymised by the publishers); `Time`
  and `Amount` are original; `Class` is the label (1 = fraud).

> Because the features are PCA outputs from one 2-day 2013 window, absolute
> scores won't transfer to another portfolio — the **methodology** is the
> transferable part. No business cost/savings figures are claimed.

## Setup

```bash
git clone https://github.com/pypi-ahmad/payment-fraud-risk-scoring.git
cd payment-fraud-risk-scoring
```


Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.12.10 (uv installs the
interpreter automatically). From the project root:

```bash
# Create the venv and install everything pinned in pyproject.toml / uv.lock
uv sync
```

### Kaggle credentials (for the dataset download)

`data/raw/creditcard.csv` is fetched automatically via `kagglehub` on first run.
That needs Kaggle credentials in **one** of these forms:

```bash
# Option A — API token env var (used by this project):
export KAGGLE_API_TOKEN=KGAT_xxxxxxxxxxxxxxxxxxxxxxxx

# Option B — classic credentials file:
mkdir -p ~/.kaggle
echo '{"username":"<user>","key":"<key>"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

If the file already exists at `data/raw/creditcard.csv`, no download or
credentials are needed — the project runs fully offline.

## How to run

### Notebook (main deliverable)

```bash
uv run jupyter lab fraud_risk_scoring.ipynb
# select the "Python 3 (payment-fraud)" kernel
```

Or run it headless end-to-end:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace fraud_risk_scoring.ipynb
```

Running the notebook trains the models and saves the chosen one to
`models/fraud_scorer.joblib`.

### Batch scoring (CLI)

```bash
uv run python batch_score.py --input data/raw/creditcard.csv --output reports/scored.csv
```

### Optional API (FastAPI)

```bash
uv run uvicorn app:app --reload
# interactive docs at http://127.0.0.1:8000/docs ; POST a transaction to /score
```

## Key findings

From a single reproducible run (`random_state=42`; stratified 60/20/20 split;
test set held out until the end):

| Model | Val PR-AUC | Val ROC-AUC |
|---|---|---|
| **XGBoost** (selected) | **0.885** | 0.978 |
| LightGBM | 0.884 | 0.979 |
| Random Forest | 0.871 | 0.963 |
| Logistic Regression (baseline) | 0.788 | 0.979 |

- **PR-AUC separates the models** even though ROC-AUC looks uniformly high —
  exactly why ROC-AUC alone is misleading under heavy imbalance. Logistic
  Regression has a strong ROC-AUC (0.979) yet collapses on PR-AUC and precision.
- **SMOTE was tested and rejected:** SMOTE+XGBoost gave 0.884 validation PR-AUC
  vs 0.885 for the class-weighted model, so the simpler model was kept.
- **Selected model:** XGBoost with `scale_pos_weight ≈ 598`, operating threshold
  **0.126** (F2-tuned on validation, favouring recall).
- **Held-out test performance:** PR-AUC **0.823**, Recall **0.800**, Precision
  **0.884**, F1 **0.840** — confusion matrix TP=76, FP=10, FN=19, TN=56,641.
- **Review-queue (Precision@K) on test:** reviewing the top 100 riskiest
  transactions yields Precision@100 ≈ 0.77 and Recall@100 ≈ 0.81 — i.e. a
  100-item queue catches ~81% of all fraud.

> Numbers are regenerated every run; the notebook is the source of truth.

## Project layout

```
payment-fraud-risk-scoring/
├── fraud_risk_scoring.ipynb   # main deliverable (executed, with outputs)
├── src/
│   ├── data.py                # Kaggle download, loading, stratified splits
│   └── scoring.py             # metrics, threshold tuning, top-K, FraudScorer
├── batch_score.py             # CLI batch scoring from the saved model
├── app.py                     # optional lightweight FastAPI scorer
├── data/raw/creditcard.csv    # dataset (downloaded; git-ignored)
├── models/fraud_scorer.joblib # saved model bundle (created by the notebook)
├── reports/                   # scored outputs
├── pyproject.toml / uv.lock   # uv-managed environment
└── .python-version            # 3.12.10
```

## Notes on compute

Everything trains in seconds on CPU (`tree_method="hist"`), so **no GPU is
required** for this dataset. XGBoost/LightGBM can use CUDA on larger data, but it
isn't needed here and isn't forced.
