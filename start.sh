#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "${WORKDIR:-$SCRIPT_DIR}"

# AI-AGENT-REF: Set thread limits for performance optimization
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

VENV_PATH="${VENV_PATH:-venv}"
# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

python validate_env.py
exec python -u -m ai_trading.main
