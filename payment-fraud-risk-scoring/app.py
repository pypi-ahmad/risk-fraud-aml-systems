"""Optional lightweight FastAPI service for real-time fraud scoring.

Serves the same model bundle the notebook produces, so online and batch scoring
are guaranteed identical. Intentionally minimal — one model, two endpoints.

Run
---
    uv run uvicorn app:app --reload

Then open http://127.0.0.1:8000/docs for an interactive form, or POST JSON:

    curl -X POST http://127.0.0.1:8000/score \
         -H "Content-Type: application/json" \
         -d '{"Time": 0, "V1": -1.36, ..., "V28": -0.02, "Amount": 149.62}'
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.scoring import FraudScorer

MODEL_PATH = Path("models/fraud_scorer.joblib")

app = FastAPI(
    title="Payment Fraud Risk Scoring",
    description="Scores a card transaction and returns its fraud risk.",
    version="0.1.0",
)

# Load once at import time; fail fast with a clear message if the model is absent.
if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model bundle not found at {MODEL_PATH}. "
        "Run fraud_risk_scoring.ipynb first to train and save it."
    )
_scorer = FraudScorer.load(MODEL_PATH)


class Transaction(BaseModel):
    """One transaction. Keys must match the model's training features."""

    model_config = {"extra": "allow"}  # tolerate extra keys (e.g. an id), ignore them

    Time: float = Field(..., description="Seconds since the first transaction.")
    Amount: float = Field(..., ge=0, description="Transaction amount.")
    # V1..V28 are supplied dynamically; declared via extra='allow' above.


class ScoreResponse(BaseModel):
    fraud_probability: float
    risk_score: int
    risk_band: str
    is_flagged: int
    threshold: float


@app.get("/health")
def health() -> dict:
    """Liveness check plus a snapshot of which model is loaded."""
    return {"status": "ok", "model": _scorer.metadata, "threshold": _scorer.threshold}


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction) -> ScoreResponse:
    """Score a single transaction."""
    row = pd.DataFrame([txn.model_dump()])
    try:
        scored = _scorer.score_frame(row).iloc[0]
    except ValueError as exc:  # missing feature columns
        raise HTTPException(status_code=422, detail=str(exc))

    return ScoreResponse(
        fraud_probability=float(scored["fraud_probability"]),
        risk_score=int(scored["risk_score"]),
        risk_band=str(scored["risk_band"]),
        is_flagged=int(scored["is_flagged"]),
        threshold=float(_scorer.threshold),
    )
