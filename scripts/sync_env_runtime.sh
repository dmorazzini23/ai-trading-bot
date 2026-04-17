#!/usr/bin/env bash
set -euo pipefail
cd /home/aiuser/ai-trading-bot

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "/home/aiuser/ai-trading-bot/venv/bin/python" ]]; then
    PYTHON_BIN="/home/aiuser/ai-trading-bot/venv/bin/python"
  elif command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.12)"
  else
    echo "python3.12 is required; run bash scripts/bootstrap.sh first" >&2
    exit 1
  fi
fi

if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
  echo "PYTHON_BIN must point to a Python 3.12 interpreter: ${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/runtime_env_sync.py --src .env --dst .env.runtime
