#!/usr/bin/env bash
set -euo pipefail

# AI-AGENT-REF: install runtime + dev dependencies quickly
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -x "$ROOT_DIR/venv/bin/python" ]; then
  python3 -m venv "$ROOT_DIR/venv" --system-site-packages
fi

"$ROOT_DIR/venv/bin/python" -m pip install --upgrade pip
"$ROOT_DIR/venv/bin/python" -m pip install -r "$ROOT_DIR/requirements-dev.txt" -r "$ROOT_DIR/requirements-test.txt"
"$ROOT_DIR/venv/bin/python" -m pip install -e "$ROOT_DIR" --no-deps
"$ROOT_DIR/ci/scripts/verify_alpaca_sdk.sh"

echo "[bootstrap] dependencies installed" >&2
