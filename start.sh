#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "${WORKDIR:-$SCRIPT_DIR}"

VENV_PATH="${VENV_PATH:-venv}"
# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

python validate_env.py
exec python -u -m ai_trading.main
