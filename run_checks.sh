#!/usr/bin/env bash
set -euo pipefail

# AI-AGENT-REF: install runtime and dev dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
ci/scripts/forbid_alpaca_trade_api.sh
ci/scripts/forbid_logging_warn.sh
python tools/ci_guard_no_apca.py

# AI-AGENT-REF: lint and run tests
ruff --select E9,F63,F7,F82,BLE001,DTZ005 --force-exclude . || true
pytest -n auto --disable-warnings -q

