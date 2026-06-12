# Transaction Anomaly Detection (Unsupervised)

A clean, practical project for detecting unusual transaction behaviour
with **mostly unsupervised** methods. Labels (fraud / not-fraud) are treated as
a *scoring yardstick*, not as training signal.

## Problem
Fraud is rare (~0.17% here) and labels are expensive, late, or incomplete.
Instead of a supervised classifier, we rank every transaction by an anomaly
score — *how unlike normal behaviour is this?* — and then, where labels exist,
measure how well that ranking surfaces real fraud.

## Dataset
**Kaggle Credit Card Fraud Detection** (`mlg-ulb/creditcardfraud`):
284,807 European card transactions over two days, 492 of them fraudulent
(0.173%). Features `V1..V28` are anonymised PCA components; only `Time`
(seconds since the first transaction) and `Amount` are in original units;
`Class` is the fraud label.

The data loader (`src/data.py`) tries, in order: a local
`data/creditcard.csv` → a `kagglehub` download → a **realistic synthetic
stream**, so the notebook always runs even without Kaggle access.

## Setup

```bash
git clone https://github.com/pypi-ahmad/transaction-anomaly-detection.git
cd transaction-anomaly-detection
```

Requires [uv](https://docs.astral.sh/uv/). Python 3.12.10 is pinned via
`.python-version` and installed automatically by uv.

```bash
cd transaction-anomaly-detection
uv sync          # creates .venv and installs all dependencies from uv.lock
```

Kaggle download (optional — only if `data/creditcard.csv` isn't already
present) needs credentials, e.g.:

```bash
export KAGGLE_USERNAME=<your-username>
export KAGGLE_KEY=<your-key>
```

## How to run
**Notebook (main deliverable):**

```bash
uv run jupyter lab transaction_anomaly_detection.ipynb
# select the "Python 3 (transaction-anomaly-detection)" kernel
```

Regenerate the notebook from source and re-execute it headless:

```bash
uv run python build_notebook.py
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 \
  transaction_anomaly_detection.ipynb
```

**Batch scoring CLI** — rank any transaction CSV by anomaly score:

```bash
uv run python -m src.score --input data/creditcard.csv \
  --output outputs/ranked.csv --top-k 100
```

## Project layout
```
transaction-anomaly-detection/
├── transaction_anomaly_detection.ipynb   # main end-to-end notebook
├── build_notebook.py                      # regenerates the notebook
├── src/
│   ├── data.py        # load Kaggle data / synthetic fallback
│   ├── features.py    # amount / time / velocity-style features
│   ├── models.py      # IsolationForest, LOF, One-Class SVM + scoring
│   ├── metrics.py     # PR-AUC, ROC-AUC, Precision@K, Recall@K
│   └── score.py       # batch scoring CLI
├── data/              # creditcard.csv (gitignored)
├── outputs/           # ranked CSVs (gitignored)
├── pyproject.toml / uv.lock
```

## Key findings
Models scored on a stratified 50k sample (all 492 frauds + random normals),
`K = #fraud`. Higher anomaly score = more anomalous; rank-normalised to `[0,1]`.

| Model | PR-AUC | ROC-AUC | Precision@K | Recall@K |
|---|---|---|---|---|
| **Isolation Forest** | **0.48** | 0.95 | 0.49 | 0.49 |
| One-Class SVM (Nystroem+SGD) | 0.22 | 0.94 | 0.31 | 0.31 |
| Local Outlier Factor | 0.04 | 0.57 | 0.07 | 0.07 |

*(Exact numbers vary slightly with the sample seed.)*

- **Isolation Forest wins**: best PR-AUC, fast, scales to the full dataset, and
  needs little tuning — the recommended detector. On the full 284k dataset its
  top-100 alerts contain ~37 frauds (**precision@100 ≈ 0.37**).
- **One-Class SVM** is mid-pack and the most tuning-sensitive (`nu`/`gamma`).
- **LOF** underperforms here and is the heaviest to compute — its local-density
  assumption doesn't fit this data well.
- **ROC-AUC is misleadingly high** for everyone because true negatives dominate;
  **PR-AUC is the honest headline** under ~600:1 imbalance.

## Key limitations
- **No account/card id**, so true per-account velocity is impossible — the
  velocity features are *stream-level* proxies and weaker than real per-entity
  signal would be.
- Models compared on a **subsample** for a fair head-to-head (LOF / OCSVM don't
  scale to 284k); results depend on the sample seed.
- This is a **learning project, not a production system**: no drift monitoring,
  no retraining schedule, no live serving. No production claims are made.
