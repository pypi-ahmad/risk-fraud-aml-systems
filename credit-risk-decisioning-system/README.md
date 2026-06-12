# credit-risk-decisioning-system

## Project goal
Build an end-to-end credit risk decisioning system for **approve / manual review / reject** policy bands, with calibrated probabilities, threshold policy controls, and deployable scoring.

## Dataset
- Competition: Home Credit Default Risk
- Link: https://www.kaggle.com/competitions/home-credit-default-risk

Download commands:
```bash
mkdir -p data/raw/home_credit
kaggle competitions download -c home-credit-default-risk -p data/raw/home_credit
unzip data/raw/home_credit/home-credit-default-risk.zip -d data/raw/home_credit
```

## Setup (uv + Python 3.12.10)

```bash
git clone https://github.com/pypi-ahmad/credit-risk-decisioning-system.git
cd credit-risk-decisioning-system
```

```bash
cd credit-risk-decisioning-system
uv venv --python 3.12.10 .venv
source .venv/bin/activate
uv sync
```

## Activate environment
```bash
cd credit-risk-decisioning-system
source .venv/bin/activate
```

## Run notebook
```bash
jupyter lab credit_risk_decisioning_system.ipynb
```

## Model-selection policy
- Track 1 uses LazyPredict discovery after leakage checks and realistic split.
- Only **top 3 eligible model families** from LazyPredict move into manual engineering.
- Eligibility filters out unstable, too-slow, badly calibrated, or low-interpretability options.

## FLAML optimization workflow
- Uses explicit `time_budget` and PR-AUC-oriented optimization.
- Reviews estimator search trace, best estimator, and cost/performance tradeoff.
- Compares FLAML winner against manual and PyCaret tracks for production fit.

## PyCaret experiment-lab workflow
- Uses `setup`, `compare_models`, `tune_model`, `calibrate_model`, `finalize_model`, and `save_model`.
- Evaluates whether calibration changes approve/review/reject policy materially.
- Retains deployable finalized artifact only if business ranking improves.

## Artifacts produced
- `artifacts/leaderboard_credit_risk.csv`
- `artifacts/credit_risk_model.joblib`
- `artifacts/credit_risk_preprocessor.joblib`
- `artifacts/decision_thresholds.json`
- optional PyCaret model artifact prefix in `artifacts/`

## Deployment and monitoring notes
- FastAPI scorer in `app/main.py` serves probability + decision band.
- Monitoring should track PR-AUC, Brier score drift, approval-rate drift, score PSI, and rejection overrides.
- Retraining trigger example: PR-AUC degradation > 10% or PSI > 0.2 for 2 consecutive weeks.

## Helper scripts
- `scripts/setup_env.sh`
- `scripts/download_data.sh`
- `scripts/run_notebook.sh`
