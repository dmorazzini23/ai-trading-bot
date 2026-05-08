#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${REPO_ROOT}/venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/venv/bin/python"
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

ENV_SRC="${AI_TRADING_ENV_SRC:-.env}"
if [[ ! -f "${ENV_SRC}" ]]; then
  echo "AI_TRADING_ENV_SRC does not exist: ${ENV_SRC}" >&2
  exit 1
fi
if [[ -n "${AI_TRADING_RUNTIME_ENV_DST:-}" ]]; then
  RUNTIME_ENV_DST="${AI_TRADING_RUNTIME_ENV_DST}"
elif [[ -d "/run/ai-trading-bot" && -w "/run/ai-trading-bot" ]]; then
  RUNTIME_ENV_DST="/run/ai-trading-bot/ai-trading-runtime.env"
else
  RUNTIME_ENV_DST="runtime/ai-trading-runtime.env"
fi
mkdir -p "$(dirname "${RUNTIME_ENV_DST}")"

"${PYTHON_BIN}" scripts/runtime_env_sync.py --src "${ENV_SRC}" --dst "${RUNTIME_ENV_DST}"
