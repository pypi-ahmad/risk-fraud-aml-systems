from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "credit_risk_model.joblib"
PREPROCESSOR_PATH = ARTIFACT_DIR / "credit_risk_preprocessor.joblib"
THRESHOLD_PATH = ARTIFACT_DIR / "decision_thresholds.json"

app = FastAPI(title="Credit Risk Decisioning API", version="0.1.0")


class ScoreRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ..., description="Single applicant payload as a flat feature dictionary"
    )


def _load_thresholds() -> tuple[float, float]:
    if THRESHOLD_PATH.exists():
        with THRESHOLD_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("approve_threshold", 0.25)), float(data.get("reject_threshold", 0.65))
    return 0.25, 0.65


def _decision_band(score: float, approve_thr: float, reject_thr: float) -> str:
    if score < approve_thr:
        return "approve"
    if score >= reject_thr:
        return "reject"
    return "manual_review"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/score")
def score(request: ScoreRequest):
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise HTTPException(status_code=404, detail="Model artifacts not found. Train notebook first.")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    payload_df = pd.DataFrame([request.features])
    x = preprocessor.transform(payload_df)
    prob_default = float(model.predict_proba(x)[:, 1][0])

    approve_thr, reject_thr = _load_thresholds()
    band = _decision_band(prob_default, approve_thr, reject_thr)

    return {
        "probability_of_default": prob_default,
        "decision_band": band,
        "thresholds": {
            "approve_threshold": approve_thr,
            "reject_threshold": reject_thr,
        },
    }
