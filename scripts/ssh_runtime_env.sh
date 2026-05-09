#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ENV_FILE="${AI_TRADING_RUNTIME_ENV_PATH:-/run/ai-trading-bot/ai-trading-runtime.env}"
if [[ ! -f "${RUNTIME_ENV_FILE}" ]]; then
  RUNTIME_ENV_FILE="${ROOT_DIR}/runtime/ai-trading-runtime.env"
fi
if [[ ! -f "${RUNTIME_ENV_FILE}" ]]; then
  RUNTIME_ENV_FILE="${ROOT_DIR}/.env"
fi
VENV_BIN="${ROOT_DIR}/venv/bin"
PYTHON_BIN="${VENV_BIN}/python"

export PATH="${VENV_BIN}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

_load_env_file() {
  local env_file="$1"
  [[ -f "${env_file}" ]] || return 0
  if [[ ! -r "${env_file}" ]]; then
    echo "runtime env file is not readable: ${env_file}" >&2
    return 1
  fi
  local exports
  exports="$("${PYTHON_BIN}" - "${env_file}" <<'PY'
from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

try:
    from dotenv import dotenv_values
except Exception:
    dotenv_values = None

ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
path = Path(sys.argv[1])

if dotenv_values is not None:
    values = dotenv_values(path)
else:
    values = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        if ENV_KEY_RE.match(key):
            values[key] = value.strip()

for key, value in values.items():
    if value is None or not ENV_KEY_RE.match(str(key)):
        continue
    print(f"export {key}={shlex.quote(str(value))}")
PY
)"
  if [[ -n "${exports}" ]]; then
    eval "${exports}"
  fi
}

# Match the installed service's writable runtime locations.
export AI_TRADING_DATA_DIR="${AI_TRADING_DATA_DIR:-/var/lib/ai-trading-bot}"
export AI_TRADING_CACHE_DIR="${AI_TRADING_CACHE_DIR:-/var/cache/ai-trading-bot}"
export AI_TRADING_LOG_DIR="${AI_TRADING_LOG_DIR:-/var/log/ai-trading-bot}"
export AI_TRADING_MODELS_DIR="${AI_TRADING_MODELS_DIR:-/var/lib/ai-trading-bot/models}"
export AI_TRADING_OUTPUT_DIR="${AI_TRADING_OUTPUT_DIR:-/var/lib/ai-trading-bot/output}"
export AI_TRADING_MODEL_PATH="${AI_TRADING_MODEL_PATH:-/var/lib/ai-trading-bot/models/trained_model.pkl}"
export AI_TRADING_MODEL_MODULE="${AI_TRADING_MODEL_MODULE:-ai_trading.simple_models}"
export TRADE_LOG_PATH="${TRADE_LOG_PATH:-/var/log/ai-trading-bot/trades.jsonl}"
export BOT_LOG_FILE="${BOT_LOG_FILE:-/var/log/ai-trading-bot/bot.log}"
export API_PORT="${API_PORT:-9001}"

export HOME="${HOME:-/var/lib/ai-trading-bot}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/var/cache/ai-trading-bot/.cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-/var/lib/ai-trading-bot/.config}"
export TMPDIR="${TMPDIR:-/var/cache/ai-trading-bot/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/var/cache/ai-trading-bot/pip}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/var/cache/ai-trading-bot/numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/var/lib/ai-trading-bot/.config/matplotlib}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/var/cache/ai-trading-bot/hf}"
export HF_HOME="${HF_HOME:-/var/cache/ai-trading-bot/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/var/cache/ai-trading-bot/hf}"

mkdir -p \
  "${AI_TRADING_DATA_DIR}" \
  "${AI_TRADING_CACHE_DIR}" \
  "${AI_TRADING_LOG_DIR}" \
  "${AI_TRADING_MODELS_DIR}" \
  "${AI_TRADING_OUTPUT_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${TMPDIR}" \
  "${PIP_CACHE_DIR}" \
  "${NUMBA_CACHE_DIR}" \
  "${MPLCONFIGDIR}" \
  "${HUGGINGFACE_HUB_CACHE}"

_load_env_file "${RUNTIME_ENV_FILE}"

if [[ "${API_PORT:-}" == "${HEALTHCHECK_PORT:-}" && "${RUN_HEALTHCHECK:-0}" == "1" ]]; then
  echo "RUN_HEALTHCHECK=1 requires HEALTHCHECK_PORT to differ from API_PORT" >&2
  exit 1
fi

# Mirror the later systemd model override drop-in after runtime env is loaded.
export AI_TRADING_MODEL_PATH="/var/lib/ai-trading-bot/models/trained_model.pkl"
export AI_TRADING_MODEL_MODULE="ai_trading.simple_models"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

cd "${ROOT_DIR}"

if [[ $# -gt 0 ]]; then
  exec "$@"
fi

exec "${SHELL:-/bin/bash}" -l
