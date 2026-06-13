# Security Policy

## Supported Versions

This repository currently supports the latest `main` branch state.

## Reporting a Vulnerability

Please do **not** open public issues for security vulnerabilities.

Send details to the maintainer with:

- vulnerability description,
- impact assessment,
- reproduction steps or proof of concept,
- suggested remediation (if available).

You will receive acknowledgement and triage updates.

## Security Notes for Users

- Treat model artifacts (`*.joblib`) as executable trust boundaries.
- Prefer loading artifacts with integrity sidecars (`*.sha256`).
- Keep API endpoints behind authentication and network controls in production.
- Store credentials only in secure environment variables or secret stores.
