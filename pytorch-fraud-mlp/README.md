# PyTorch Fraud Detection — MLP Classifier

A compact, interview-ready PyTorch project for **binary fraud detection on severely imbalanced tabular data**.

## Task

Binary classification: given a credit card transaction, is it fraudulent?
Core challenge: class imbalance (~0.17% fraud rate).

## Dataset

**[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**
by the Machine Learning Group — Université Libre de Bruxelles (ULB)

| Property | Value |
|----------|-------|
| Transactions | 284,807 |
| Fraud cases | 492 (0.172%) |
| Features | V1–V28 (PCA-anonymised), Time (s), Amount (€) |
| Label | `Class` — 0 normal, 1 fraud |
| Source | Kaggle: `mlg-ulb/creditcardfraud` |

## Model Architecture

```
Input (30 features)
  └─ Linear(30→256) → BatchNorm1d → GELU → Dropout(0.3)
       └─ Linear(256→128) → BatchNorm1d → GELU → Dropout(0.3)
            └─ Linear(128→64) → BatchNorm1d → GELU → Dropout(0.3)
                 └─ Linear(64→1)  ← raw logit
```

- **Parameters**: ~50 K
- **Activation**: GELU
- **Normalisation**: BatchNorm1d after each linear layer
- **Regularisation**: Dropout(0.3)
- **Initialisation**: Kaiming normal

## Imbalance Strategy

The notebook **tests** imbalance techniques rather than assuming them. An ablation
compares `pos_weight` ∈ {1, √ratio, ratio/4, ratio} and shows that aggressive
weighted loss *lowers* PR-AUC and saturates probabilities (F1-optimal threshold → ~1.0).

| Lever | Used? | Note |
|-------|-------|------|
| Threshold tuning (max F1 on **validation**) | ✓ Primary | Tuned per-model; applied to test |
| PR-AUC for model selection & evaluation | ✓ | Imbalance-aware metric |
| `BCEWithLogitsLoss(pos_weight)` | ✗ (ablated) | Hurt PR-AUC on this dataset |
| `WeightedRandomSampler` | ✗ | Over-corrects when stacked with weighted loss |

**Winning recipe:** plain `BCEWithLogitsLoss` + validation threshold tuning.

## Training Procedure

| Hyperparameter | Value |
|----------------|-------|
| Loss | `BCEWithLogitsLoss` (plain — see ablation) |
| Optimiser | AdamW (fused CUDA kernel) |
| Learning rate | 1e-3 → 1e-6 (CosineAnnealingLR) |
| Weight decay | 1e-4 |
| Batch size | 512 (shuffled, `drop_last=True`) |
| Max epochs | 60 |
| Early stopping | Patience 10 on val PR-AUC |
| Grad clipping | max_norm=1.0 |
| Checkpoint | Best val PR-AUC saved to `checkpoints/best_model.pt` |

## Key Metrics (Test Set)

Primary metric: **PR-AUC (Average Precision)** — more meaningful than ROC-AUC on imbalanced data.
Thresholds are tuned per-model on the validation set, then applied to test.

| Model | PR-AUC | ROC-AUC | F1 (tuned) | Recall | Precision |
|-------|--------|---------|------------|--------|-----------|
| Logistic Regression | ~0.67 | ~0.96 | ~0.78 | ~0.77 | ~0.79 |
| Random Forest | ~0.82 | ~0.97 | ~0.80 | ~0.74 | ~0.86 |
| **MLP (PyTorch)** | **~0.84** | **~0.97** | **~0.82** | **~0.81** | **~0.82** |

On the test set the MLP catches ~60/74 fraud cases with ~13 false positives.
*Exact numbers vary slightly by run; seed = 42.*

## Setup

```bash
git clone https://github.com/pypi-ahmad/pytorch-fraud-mlp.git
cd pytorch-fraud-mlp
```


### Requirements

- Ubuntu / Linux
- NVIDIA GPU with CUDA driver ≥ 12.8 (CUDA 13.x driver is compatible)
- [uv](https://github.com/astral-sh/uv) (Python package manager)

### 1 — Clone / enter project

```bash
cd pytorch-fraud-mlp
```

### 2 — Install dependencies

```bash
uv sync --python 3.12.10
```

This creates `.venv/` with Python 3.12.10 and installs all packages including
`torch==2.11.0+cu128`.

### 3 — Set Kaggle credentials

```bash
mkdir -p ~/.kaggle
echo 'YOUR_KAGGLE_TOKEN' > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

Or export the environment variable:

```bash
export KAGGLE_TOKEN=YOUR_KAGGLE_TOKEN
```

### 4 — Generate the notebook

```bash
uv run python generate_notebook.py
```

This creates `pytorch_fraud_mlp.ipynb`.

### 5 — Register the kernel

```bash
uv run python -m ipykernel install --user --name pytorch-fraud-mlp --display-name "pytorch-fraud-mlp"
```

### 6 — Launch Jupyter

```bash
uv run jupyter notebook pytorch_fraud_mlp.ipynb
```

Or Jupyter Lab:

```bash
uv run jupyter lab pytorch_fraud_mlp.ipynb
```

The notebook downloads the dataset automatically on first run.

## Inference on New Data

```bash
uv run python inference.py --input new_transactions.csv --output predictions.csv --threshold 0.35
```

The input CSV must have the same schema as the original dataset (V1–V28, Time, Amount).

## Project Structure

```
pytorch-fraud-mlp/
├── pyproject.toml            # uv project + PyTorch CUDA index
├── generate_notebook.py      # Generates the .ipynb
├── inference.py              # CLI inference helper
├── README.md
├── pytorch_fraud_mlp.ipynb   # Main notebook (generated)
├── data/
│   └── creditcard.csv        # Auto-downloaded from Kaggle
├── checkpoints/
│   └── best_model.pt         # Saved during training
└── figures/
    ├── eda_overview.png
    ├── training_curves.png
    ├── pr_and_threshold.png
    ├── confusion_matrix.png
    ├── model_comparison.png
    └── pr_curve_comparison.png
```

## Environment

| Tool | Version |
|------|---------|
| Python | 3.12.10 |
| PyTorch | 2.11.0+cu128 |
| CUDA target | 12.8 (compatible with driver 13.x) |
| scikit-learn | 1.9.0 |
| pandas | 2.2.x |
| uv | 0.11.19 |
