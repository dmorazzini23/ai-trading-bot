#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "${WORKDIR:-$SCRIPT_DIR}"

# AI-AGENT-REF: Set thread limits for performance optimization
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export AI_TRADING_MODEL_MODULE="${AI_TRADING_MODEL_MODULE:-ai_trading.model_loader}"

# Ensure position sizing limit is defined before launching
if [ -z "${AI_TRADING_MAX_POSITION_SIZE:-}" ]; then
  echo "AI_TRADING_MAX_POSITION_SIZE is required" >&2
  exit 1
fi

VENV_PATH="${VENV_PATH:-venv}"
# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

python -m ai_trading.tools.env_validate
exec python -u -m ai_trading.main
