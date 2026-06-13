# Contributing

Thanks for contributing to `payment-fraud-risk-scoring`.

## Development Setup

```bash
uv venv --python 3.12.10 .venv
source .venv/bin/activate
uv sync
```

## Common Commands

```bash
# run tests
uv run python -m unittest discover -s tests -v

# run API locally
uv run uvicorn app:app --reload

# run batch scoring
uv run python batch_score.py --input data/raw/creditcard.csv --output reports/scored.csv
```

## Pull Request Guidelines

1. Keep changes scoped and maintainable.
2. Add or update tests for behavioral changes.
3. Update README/docs for any API or CLI contract changes.
4. Never commit secrets, tokens, or local credential files.

## Reporting Bugs

Please include:

- expected behavior,
- observed behavior,
- reproduction steps,
- environment details (`python`, `uv`, OS),
- relevant logs/tracebacks.
