# Changelog

All notable changes to this project are documented in this file.

## [0.3.0] - 2026-06-13

### Added

- Integrity sidecar (`.sha256`) support for saved model artifacts.
- API key gate (`FRAUD_API_KEY`) for `/score` and `/admin/reload-model`.
- `/admin/reload-model` endpoint for model reload without process restart.
- `tests/` suite for scoring, API, and batch scoring paths.
- OSS governance files: LICENSE, CONTRIBUTING, SECURITY, CODE_OF_CONDUCT.
- reusable training pipeline in `src/training.py` plus `train_model.py` CLI.
- per-client in-memory rate limiting for API scoring (`FRAUD_RATE_LIMIT_PER_MIN`).

### Changed

- Strict request schema for API scoring (explicit `Time`, `V1..V28`, `Amount`).
- API startup now degrades gracefully when model is missing, instead of crashing.
- Batch scoring supports optional chunked processing and empty-file-safe output.
- Scoring utilities now validate edge cases and expose clearer errors.
- Model hash sidecar verification is enforced by default (`FRAUD_REQUIRE_MODEL_HASH=true`).

### Fixed

- Batch metadata access no longer fails when scorer metadata is `None`.
- Division-by-zero risk removed for empty batch files.
- ROC-AUC edge cases with single-class inputs handled safely.
