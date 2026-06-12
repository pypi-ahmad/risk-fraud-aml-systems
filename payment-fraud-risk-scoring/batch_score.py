"""Batch fraud scoring from the command line.

Loads the model bundle saved by the notebook (``models/fraud_scorer.joblib``)
and scores every row of an input CSV, writing out fraud probabilities, integer
risk scores, risk bands and the binary flag decision.

Usage
-----
    uv run python batch_score.py --input data/raw/creditcard.csv --output reports/scored.csv

The input CSV must contain the feature columns the model was trained on
(``Time``, ``V1``..``V28``, ``Amount``). A ``Class`` column, if present, is
ignored for scoring and passed through untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.scoring import FraudScorer

DEFAULT_MODEL = "models/fraud_scorer.joblib"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch-score transactions for fraud risk.")
    p.add_argument("--input", required=True, help="Path to input CSV of transactions.")
    p.add_argument("--output", required=True, help="Path to write the scored CSV.")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Model bundle (default: {DEFAULT_MODEL}).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            f"Model bundle not found at {model_path}. "
            "Run fraud_risk_scoring.ipynb first to train and save it."
        )

    scorer = FraudScorer.load(model_path)
    df = pd.read_csv(args.input)

    scored = scorer.score_frame(df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_path, index=False)

    flagged = int(scored["is_flagged"].sum())
    print(f"Model     : {scorer.metadata.get('model_name', 'unknown')} "
          f"(threshold {scorer.threshold:.4f})")
    print(f"Scored    : {len(scored):,} transactions")
    print(f"Flagged   : {flagged:,} ({flagged / len(scored):.4%}) for review")
    print(f"Written to: {out_path}")


if __name__ == "__main__":
    main()
