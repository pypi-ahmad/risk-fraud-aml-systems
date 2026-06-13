# Payment Fraud Risk Scoring

[![Python](https://img.shields.io/badge/python-3.12.10-blue.svg)](https://www.python.org/downloads/release/python-31210/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

Production-oriented fraud risk scoring project on an extremely imbalanced card-transaction dataset (~0.17% fraud). The system supports both notebook and script-driven training and exposes reusable inference surfaces via:

- a serializable `FraudScorer` bundle,
- a batch CSV scoring CLI,
- an optional FastAPI service.
- a scriptable training CLI (`train_model.py`) for non-notebook workflows.

## Why This Project

In card fraud, class imbalance makes plain accuracy misleading. This project optimizes and reports fraud-relevant metrics:

- **PR-AUC** for model selection under heavy imbalance,
- **Recall** to minimize expensive missed fraud,
- **Precision/F1** to control analyst review load,
- **Top-K review metrics** (`precision@k`, `recall@k`) for queue operations.

Thresholding is explicit and tuned on validation; it is not fixed at `0.5`.

## Dataset

- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Kaggle slug: `mlg-ulb/creditcardfraud`
- Period: two days of European card activity (September 2013)
- Raw class mix: 492 frauds out of 284,807 transactions (0.173%)
- This project drops fully duplicated rows before splitting.

## Repository Layout

```text
payment-fraud-risk-scoring/
├── fraud_risk_scoring.ipynb     # main training/evaluation notebook
├── train_model.py               # scriptable training entrypoint
├── src/
│   ├── data.py                  # data acquisition/loading/splitting
│   └── scoring.py               # metrics, thresholding, inference bundle
├── app.py                       # optional FastAPI scoring service
├── batch_score.py               # batch CSV scoring CLI
├── tests/                       # unit/integration-style tests
├── reports/                     # generated scored outputs
├── pyproject.toml
├── uv.lock
└── README.md
```

## Quickstart

### 1. Clone and enter

```bash
git clone https://github.com/pypi-ahmad/payment-fraud-risk-scoring.git
cd payment-fraud-risk-scoring
```

### 2. Create environment (Python 3.12.10 via `uv`)

```bash
uv venv --python 3.12.10 .venv
source .venv/bin/activate
uv sync
```

### 3. Configure Kaggle access (first download only)

The project attempts dataset download via `kagglehub` if `data/raw/creditcard.csv` is missing.

```bash
# Option A: Kaggle API token env var (project-documented path)
export KAGGLE_API_TOKEN=KGAT_xxxxxxxxxxxxxxxxxxxxxxxx

# Option B: ~/.kaggle/kaggle.json
mkdir -p ~/.kaggle
echo '{"username":"<user>","key":"<key>"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

If `data/raw/creditcard.csv` already exists, all runs are offline.

## Running

### Scripted training (recommended for CI/automation)

```bash
uv run python train_model.py \
  --output-model models/fraud_scorer.joblib \
  --metrics-out reports/training_metrics.json
```

This writes:

- `models/fraud_scorer.joblib`
- `models/fraud_scorer.joblib.sha256`
- `reports/training_metrics.json`

### Notebook training/evaluation

```bash
uv run jupyter lab fraud_risk_scoring.ipynb
```

Headless execution:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace fraud_risk_scoring.ipynb
```

Notebook execution writes `models/fraud_scorer.joblib` and `models/fraud_scorer.joblib.sha256`.

### Batch scoring (CLI)

```bash
uv run python batch_score.py \
  --input data/raw/creditcard.csv \
  --output reports/scored.csv
```

For large CSVs:

```bash
uv run python batch_score.py \
  --input data/raw/creditcard.csv \
  --output reports/scored.csv \
  --chunksize 50000
```

### API scoring (FastAPI)

```bash
uv run uvicorn app:app --reload
```

OpenAPI docs: `http://127.0.0.1:8000/docs`

Optional API key protection:

```bash
export FRAUD_API_KEY="replace-with-strong-secret"
```

When configured, send `X-API-Key` for `POST /score` and `POST /admin/reload-model`.

Optional per-client rate limit (default: 120 requests/min):

```bash
export FRAUD_RATE_LIMIT_PER_MIN=120
```

Model integrity policy (default requires hash sidecar):

```bash
export FRAUD_REQUIRE_MODEL_HASH=true
```

## API Contract

### `GET /health`

Returns service/model readiness.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "/abs/path/models/fraud_scorer.joblib",
  "rate_limit_per_min": 120
}
```

### `POST /score`

Request body must contain all model features:

- `Time` (float)
- `V1` ... `V28` (float)
- `Amount` (float, `>= 0`)

Unknown fields are rejected.
Requests are rate-limited per client key (IP / `x-forwarded-for`) based on `FRAUD_RATE_LIMIT_PER_MIN`.

Example response:

```json
{
  "fraud_probability": 0.8421,
  "risk_score": 842,
  "risk_band": "High",
  "is_flagged": 1,
  "threshold": 0.126
}
```

### `POST /admin/reload-model`

Reloads the model artifact from disk (useful after retraining without process restart).

## Model Card (Short)

- **Intended use**: transaction risk ranking and analyst queue prioritization.
- **Primary metric**: PR-AUC on validation.
- **Operating policy**: explicit threshold tuning (e.g., F2-optimized threshold).
- **Not intended for**: direct business-loss estimates, policy transfer across unseen portfolios without recalibration.
- **Data limitations**: anonymized PCA feature space from a fixed 2013 2-day window.

## Reproducibility Notes

- Notebook sets deterministic seed (`random_state=42`) for data split/model training.
- `train_model.py` uses the same deterministic seed and model-selection policy.
- Train/validation/test splitting is stratified and performed before fitting.
- Lockfile-managed dependencies are in `uv.lock`.

## Testing

Run the local test suite:

```bash
uv run python -m unittest discover -s tests -v
```

Current tests cover:

- scoring/threshold edge cases,
- model artifact hash verification,
- API readiness/auth paths,
- batch scoring behavior for normal and empty inputs,
- scripted training pipeline artifact generation.

## Security

- Never commit Kaggle credentials or API keys.
- Model loading supports SHA-256 integrity sidecars (`*.sha256`).
- If deploying publicly, place API behind network controls, TLS, auth, and rate limiting.

Report vulnerabilities using [SECURITY.md](./SECURITY.md).

## Documentation and Governance

- [CONTRIBUTING.md](./CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)
- [SECURITY.md](./SECURITY.md)
- [CHANGELOG.md](./CHANGELOG.md)

## License

Licensed under the [MIT License](./LICENSE).

## Troubleshooting

- **`Model bundle not found`**: execute notebook first to produce `models/fraud_scorer.joblib`.
- **`Failed to download dataset`**: verify Kaggle credentials/network, or place CSV at `data/raw/creditcard.csv`.
- **`Model hash mismatch`**: artifact was modified after save; regenerate via notebook training.
- **`/score` returns `503`**: model not loaded; check `/health` and run `/admin/reload-model` after fixing artifact path.
