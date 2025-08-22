#!/usr/bin/env bash
set -euo pipefail

# AI-AGENT-REF: install runtime and dev dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt

# AI-AGENT-REF: lint and run tests
ruff --select E9,F63,F7,F82,BLE001,DTZ005 --force-exclude . || true
pytest -n auto --disable-warnings -q

