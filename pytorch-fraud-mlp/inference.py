#!/usr/bin/env python
"""
inference.py — Run fraud prediction on a new CSV file.

Usage:
    uv run python inference.py --input transactions.csv
    uv run python inference.py --input transactions.csv --output preds.csv --threshold 0.35

Input CSV must have the same schema as the training data:
    V1, V2, ..., V28, Time, Amount  (Class column is optional / ignored)

The checkpoint is loaded from checkpoints/best_model.pt by default.

NOTE: This re-standardises Amount and Time using the statistics of the INPUT
file itself. For production, serialise the training-set scalers (e.g. joblib)
and load them here instead.
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Model (must match training definition)
# ─────────────────────────────────────────────────────────────────────────────

class FraudMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.3):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers: List[nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), nn.BatchNorm1d(out_d), nn.GELU(), nn.Dropout(p=dropout)]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount_s", "Time_s"]


def load_model(ckpt_path: Path, device: torch.device) -> tuple:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m    = FraudMLP(
        input_dim   = ckpt.get("input_dim", len(FEATURE_COLS)),
        hidden_dims = ckpt["hidden_dims"],
        dropout     = ckpt["dropout"],
    ).to(device)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m, ckpt


def preprocess(df: pd.DataFrame) -> np.ndarray:
    df = df.copy()
    # Standardise raw features (mean/std of this file — swap for saved scalers in production)
    df["Amount_s"] = (df["Amount"] - df["Amount"].mean()) / (df["Amount"].std() + 1e-9)
    df["Time_s"]   = (df["Time"]   - df["Time"].mean())   / (df["Time"].std()   + 1e-9)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")
    return df[FEATURE_COLS].to_numpy(dtype=np.float32)


@torch.no_grad()
def run_inference(
    model:     nn.Module,
    X:         np.ndarray,
    device:    torch.device,
    threshold: float,
    batch:     int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_list = []
    for start in range(0, len(X), batch):
        chunk  = torch.from_numpy(X[start : start + batch]).to(device)
        logits = model(chunk)
        probs_list.append(logits.sigmoid().cpu().numpy().ravel())
    probs = np.concatenate(probs_list)
    preds = (probs >= threshold).astype(int)
    return probs, preds


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run trained FraudMLP on new transactions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",      required=True,              help="Input CSV path")
    parser.add_argument("--output",     default="predictions.csv",  help="Output CSV path")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--threshold",  type=float, default=0.5,
                        help="Decision threshold (tune from notebook's best_thr)")
    parser.add_argument("--device",     default="auto",
                        choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device : {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}\nTrain the model first by running the notebook.")
    model, ckpt = load_model(ckpt_path, device)
    print(f"Model  : epoch {ckpt['epoch']}  val PR-AUC={ckpt['val_pr_auc']:.4f}")

    # ── Load and preprocess input ─────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Input  : {input_path}  ({len(df):,} rows)")

    X = preprocess(df)

    # ── Inference ─────────────────────────────────────────────────────────────
    probs, preds = run_inference(model, X, device, threshold=args.threshold)

    # ── Save output ───────────────────────────────────────────────────────────
    out_df = df.copy()
    out_df["fraud_probability"] = probs.round(6)
    out_df["fraud_prediction"]  = preds
    out_df.to_csv(args.output, index=False)

    n_fraud = int(preds.sum())
    print(f"\nResults:")
    print(f"  Threshold  : {args.threshold}")
    print(f"  Flagged    : {n_fraud:,} / {len(preds):,}  ({n_fraud / len(preds) * 100:.3f}%)")
    print(f"  Output     : {args.output}")


if __name__ == "__main__":
    main()
