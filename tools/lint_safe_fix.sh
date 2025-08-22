#!/usr/bin/env bash
set -euo pipefail

# AI-AGENT-REF: phase2b lint and test runner
mkdir -p artifacts

# Baseline
ruff --version | tee artifacts/ruff-version.txt
ruff check . --statistics | tee artifacts/ruff-phase2b-baseline.txt || true

# Pass 1: imports & top-of-file
ruff check . --select I,E402 --fix || true

# Pass 2: unuseds & None-comparisons
ruff check . --select F401,F403,F405,F841,E711,E712 --fix || true
python tools/codemods/none_comparison_fix.py || true

# Pass 3: report DTZ/T201 needed manual spots (no auto-fix)
ruff check . --select DTZ,T201 | tee artifacts/ruff-phase2b-dtz-t201.txt || true

# Final stats
ruff check . --statistics | tee artifacts/ruff-phase2b-final.txt || true

# Typecheck & tests
python -m mypy --version | tee artifacts/mypy-version.txt
python -m mypy ai_trading trade_execution | tee artifacts/mypy-phase2b.txt || true
pytest -n auto --disable-warnings --maxfail=0 -q | tee artifacts/pytest-phase2b.txt || true
