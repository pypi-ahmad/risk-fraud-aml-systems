#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p data/raw/home_credit
kaggle competitions download -c home-credit-default-risk -p data/raw/home_credit
unzip -o data/raw/home_credit/home-credit-default-risk.zip -d data/raw/home_credit
