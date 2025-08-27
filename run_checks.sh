#!/usr/bin/env bash
set -euo pipefail

# AI-AGENT-REF: install runtime and dev dependencies
python -m pip uninstall -y alpaca-trade-api || true
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
ci/scripts/forbid_alpaca_trade_api.sh

# AI-AGENT-REF: lint and run tests
ruff --select E9,F63,F7,F82,BLE001,DTZ005 --force-exclude . || true
pytest -n auto --disable-warnings -q

