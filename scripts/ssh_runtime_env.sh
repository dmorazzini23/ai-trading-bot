#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ENV_FILE="${ROOT_DIR}/.env.runtime"
VENV_BIN="${ROOT_DIR}/venv/bin"

export PATH="${VENV_BIN}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"

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

if [[ -f "${RUNTIME_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${RUNTIME_ENV_FILE}"
  set +a
fi

# Mirror the later systemd model override drop-in after .env.runtime is loaded.
export AI_TRADING_MODEL_PATH="/var/lib/ai-trading-bot/models/trained_model.pkl"
export AI_TRADING_MODEL_MODULE="ai_trading.simple_models"

cd "${ROOT_DIR}"

if [[ $# -gt 0 ]]; then
  exec "$@"
fi

exec "${SHELL:-/bin/bash}" -l
