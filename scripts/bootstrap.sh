#!/usr/bin/env bash
set -euo pipefail

# AI-AGENT-REF: install runtime + dev dependencies quickly
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python -m pip uninstall -y alpaca-trade-api || true
python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/requirements-dev.txt"
"$ROOT_DIR/ci/scripts/forbid_alpaca_trade_api.sh"

echo "[bootstrap] dependencies installed" >&2
