#!/usr/bin/env bash
set -euo pipefail

# AI-AGENT-REF: install runtime + dev dependencies quickly
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/requirements-dev.txt"

echo "[bootstrap] dependencies installed" >&2
