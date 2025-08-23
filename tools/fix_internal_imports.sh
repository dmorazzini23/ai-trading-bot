#!/usr/bin/env bash
set -euo pipefail
mkdir -p artifacts
python -m tools.repair_test_imports --pkg ai_trading --tests tests --write --report artifacts/import-repair-report.md
