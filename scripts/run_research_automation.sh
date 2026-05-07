#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CADENCE="${1:-daily}"
WORKFLOW="${2:-}"

PYTHON_BIN="${AI_TRADING_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${ROOT_DIR}/venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || true)"
  fi
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "missing python executable; set AI_TRADING_PYTHON or create ${ROOT_DIR}/venv/bin/python" >&2
  exit 1
fi

PY_VERSION="$("${PYTHON_BIN}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
if [[ "${PY_VERSION}" != "3.12" ]]; then
  echo "unsupported python version ${PY_VERSION}; expected 3.12" >&2
  exit 1
fi

RUNTIME_DIR="${AI_TRADING_RESEARCH_LOCK_DIR:-/var/lib/ai-trading-bot/runtime}"
mkdir -p "${RUNTIME_DIR}"
LOCK_PATH="${RUNTIME_DIR}/research_automation_${CADENCE}.lock"

ARGS=("${CADENCE}")
if [[ "${CADENCE}" == "manual" && -n "${WORKFLOW}" ]]; then
  ARGS+=("--workflow" "${WORKFLOW}")
fi
if [[ "${AI_TRADING_RESEARCH_PLAN_ONLY:-0}" == "1" ]]; then
  ARGS+=("--plan-only")
fi
if [[ "${AI_TRADING_RESEARCH_DRY_RUN:-0}" == "1" ]]; then
  ARGS+=("--dry-run")
fi

set +e
flock -n "${LOCK_PATH}" \
  "${PYTHON_BIN}" -m ai_trading.tools.research_automation "${ARGS[@]}"
STATUS=$?
set -e

if [[ "${AI_TRADING_RESEARCH_NOTIFY_SLACK:-1}" == "1" ]]; then
  if [[ "${AI_TRADING_RESEARCH_PLAN_ONLY:-0}" != "1" || "${AI_TRADING_RESEARCH_NOTIFY_PLAN_ONLY:-0}" == "1" ]]; then
    "${PYTHON_BIN}" -m ai_trading.tools.research_completion_notify \
      --cadence "${CADENCE}" \
      --workflow "${WORKFLOW:-${CADENCE}}" \
      --exit-code "${STATUS}" \
      --channel "${AI_TRADING_RESEARCH_SLACK_CHANNEL:-#all-beatwallstreet}" || true
  fi
fi

exit "${STATUS}"
