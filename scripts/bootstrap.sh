#!/usr/bin/env bash
set -euo pipefail

# AI-AGENT-REF: install runtime + dev dependencies quickly
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 -m pip install --upgrade pip
python3 -m pip install -r "$ROOT_DIR/requirements-dev.txt"
"$ROOT_DIR/ci/scripts/verify_alpaca_sdk.sh"

echo "[bootstrap] dependencies installed" >&2
