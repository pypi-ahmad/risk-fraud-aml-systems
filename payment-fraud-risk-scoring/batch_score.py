"""Batch fraud scoring from the command line.

Loads the model bundle saved by the notebook (``models/fraud_scorer.joblib``)
and scores every row of an input CSV, writing fraud probabilities, integer risk
scores, risk bands, and binary flag decisions.

Usage
-----
    uv run python batch_score.py --input data/raw/creditcard.csv --output reports/scored.csv

The input CSV must contain the feature columns the model was trained on
(``Time``, ``V1``..``V28``, ``Amount``). A ``Class`` column, if present, is
ignored for scoring and passed through untouched.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from src.scoring import FraudScorer

DEFAULT_MODEL = "models/fraud_scorer.joblib"
LOGGER = logging.getLogger("payment_fraud_batch")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("chunksize must be a positive integer.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-score transactions for fraud risk.")
    parser.add_argument("--input", required=True, help="Path to input CSV of transactions.")
    parser.add_argument("--output", required=True, help="Path to write the scored CSV.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model bundle (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--chunksize",
        type=_positive_int,
        default=None,
        help="Optional chunk size for streaming large CSV files.",
    )
    return parser.parse_args()


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def score_csv(
    input_path: Path,
    output_path: Path,
    scorer: FraudScorer,
    *,
    chunksize: int | None = None,
) -> tuple[int, int]:
    """Score an input CSV and persist the scored output.

    Returns ``(total_rows, total_flagged)``.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found at {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    total_flagged = 0
    first_write = True

    if chunksize is None:
        chunks: list[pd.DataFrame] | pd.io.parsers.TextFileReader = [pd.read_csv(input_path)]
    else:
        chunks = pd.read_csv(input_path, chunksize=chunksize)

    for chunk in chunks:
        scored_chunk = scorer.score_frame(chunk)
        scored_chunk.to_csv(
            output_path,
            mode="w" if first_write else "a",
            index=False,
            header=first_write,
        )
        first_write = False
        total_rows += len(scored_chunk)
        if not scored_chunk.empty:
            total_flagged += int(scored_chunk["is_flagged"].sum())

    # Handle truly empty files when iterator yields no rows.
    if first_write:
        empty = pd.read_csv(input_path, nrows=0)
        scored_empty = scorer.score_frame(empty)
        scored_empty.to_csv(output_path, index=False)

    return total_rows, total_flagged


def _render_summary(scorer: FraudScorer, total_rows: int, total_flagged: int, out_path: Path) -> list[str]:
    metadata = scorer.metadata or {}
    model_name = metadata.get("model_name", "unknown")
    flag_rate = (total_flagged / total_rows) if total_rows else 0.0
    return [
        f"Model     : {model_name} (threshold {scorer.threshold:.4f})",
        f"Scored    : {total_rows:,} transactions",
        f"Flagged   : {total_flagged:,} ({flag_rate:.4%}) for review",
        f"Written to: {out_path}",
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            f"Model bundle not found at {model_path}. "
            "Run fraud_risk_scoring.ipynb first to train and save it."
        )

    scorer = FraudScorer.load(
        model_path,
        verify_hash=True,
        require_hash=_bool_env("FRAUD_REQUIRE_MODEL_HASH", default=True),
    )

    input_path = Path(args.input)
    output_path = Path(args.output)

    total_rows, total_flagged = score_csv(
        input_path=input_path,
        output_path=output_path,
        scorer=scorer,
        chunksize=args.chunksize,
    )

    for line in _render_summary(scorer, total_rows, total_flagged, output_path):
        LOGGER.info(line)


if __name__ == "__main__":
    main()
