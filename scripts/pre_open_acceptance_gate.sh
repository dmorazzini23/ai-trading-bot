#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/aiuser/ai-trading-bot"
cd "${REPO_DIR}"

if [[ -f "venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "venv/bin/activate"
fi

python3 scripts/pre_open_acceptance_gate.py --json "$@"

