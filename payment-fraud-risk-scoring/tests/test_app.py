from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from fastapi import HTTPException

import app as app_module
from src.scoring import FraudScorer


class DummyModel:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        rows = len(frame)
        p = np.full(rows, 0.8, dtype=float)
        return np.column_stack((1.0 - p, p))


class _Client:
    def __init__(self, host: str) -> None:
        self.host = host


class _Request:
    def __init__(self, *, headers: dict[str, str] | None = None, host: str = "127.0.0.1") -> None:
        self.app = app_module.app
        self.headers = headers or {}
        self.client = _Client(host)


def _payload() -> dict[str, float]:
    payload: dict[str, float] = {"Time": 0.0, "Amount": 149.62}
    for idx in range(1, 29):
        payload[f"V{idx}"] = float(idx) / 10.0
    return payload


class AppTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_scorer = app_module.app.state.scorer
        self._original_path = app_module.app.state.model_path
        self._original_rate_limiter = app_module.app.state.rate_limiter

    def tearDown(self) -> None:
        app_module.app.state.scorer = self._original_scorer
        app_module.app.state.model_path = self._original_path
        app_module.app.state.rate_limiter = self._original_rate_limiter

    def test_health_reports_degraded_without_model(self) -> None:
        app_module.app.state.scorer = None
        health = app_module.health(_Request())

        self.assertEqual(health.status, "degraded")
        self.assertFalse(health.model_loaded)

    def test_score_returns_503_when_model_is_unavailable(self) -> None:
        app_module.app.state.scorer = None
        txn = app_module.Transaction(**_payload())

        with self.assertRaises(HTTPException) as context:
            app_module.score(txn, _Request())

        self.assertEqual(context.exception.status_code, 503)

    def test_auth_guard_enforces_api_key_when_configured(self) -> None:
        with mock.patch.dict(os.environ, {"FRAUD_API_KEY": "secret"}, clear=False):
            with self.assertRaises(HTTPException) as denied:
                app_module._auth_guard(None)
            self.assertEqual(denied.exception.status_code, 401)

            app_module._auth_guard("secret")

    def test_rate_limit_guard_blocks_after_limit(self) -> None:
        app_module.app.state.rate_limiter = app_module.RateLimiter(limit_per_window=1)
        request = _Request(host="10.0.0.1")
        app_module._rate_limit_guard(request)

        with self.assertRaises(HTTPException) as blocked:
            app_module._rate_limit_guard(request)

        self.assertEqual(blocked.exception.status_code, 429)

    def test_score_returns_response_payload(self) -> None:
        app_module.app.state.scorer = FraudScorer(
            model=DummyModel(),
            features=["Time", *[f"V{i}" for i in range(1, 29)], "Amount"],
            threshold=0.5,
        )
        txn = app_module.Transaction(**_payload())
        request = _Request()
        app_module._rate_limit_guard(request)
        scored = app_module.score(txn, request)

        self.assertGreater(scored.fraud_probability, 0.0)
        self.assertIn(scored.risk_band, {"Low", "Medium", "High"})
        self.assertIn(scored.is_flagged, {0, 1})


if __name__ == "__main__":
    unittest.main()
