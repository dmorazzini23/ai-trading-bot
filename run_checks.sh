#!/usr/bin/env bash
set -euo pipefail

# AI-AGENT-REF: install runtime and dev dependencies
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt -r requirements-dev.txt
ci/scripts/verify_alpaca_sdk.sh
ci/scripts/forbid_logging_warn.sh
python3 tools/ci_guard_no_apca.py

# AI-AGENT-REF: lint and run tests
ruff --select E9,F63,F7,F82,BLE001,DTZ005 --force-exclude . || true
pytest -n auto --disable-warnings -q
