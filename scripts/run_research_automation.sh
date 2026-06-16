#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUNTIME_ENV_PATH="${AI_TRADING_RUNTIME_ENV_FILE:-/run/ai-trading-bot/ai-trading-runtime.env}"
if [[ "${AI_TRADING_RESEARCH_AUTO_SOURCE_RUNTIME_ENV:-1}" == "1" \
  && -r "${RUNTIME_ENV_PATH}" \
  && -z "${AI_TRADING_SECRETS_BACKEND:-}" \
  && -z "${AI_TRADING_AWS_SECRET_ID:-}" \
  && -z "${AI_TRADING_MANAGED_SECRET_KEYS:-}" ]]; then
  set -a
  # shellcheck source=/run/ai-trading-bot/ai-trading-runtime.env
  source "${RUNTIME_ENV_PATH}"
  set +a
fi

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

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  exec "${PYTHON_BIN}" -m ai_trading.tools.research_automation --help
fi

CADENCE="${1:-daily}"
WORKFLOW="${2:-}"

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
flock -E 75 -n "${LOCK_PATH}" \
  "${PYTHON_BIN}" -m ai_trading.tools.research_automation "${ARGS[@]}"
STATUS=$?
set -e

RUN_STATUS="infrastructure_failed"
case "${STATUS}" in
  0)
    if [[ "${AI_TRADING_RESEARCH_PLAN_ONLY:-0}" == "1" ]]; then
      RUN_STATUS="planned"
    elif [[ "${AI_TRADING_RESEARCH_DRY_RUN:-0}" == "1" ]]; then
      RUN_STATUS="dry_run"
    else
      RUN_STATUS="complete"
    fi
    ;;
  2)
    RUN_STATUS="blocked"
    ;;
  75)
    RUN_STATUS="locked"
    echo "research automation ${CADENCE} run is already locked at ${LOCK_PATH}" >&2
    ;;
esac

if [[ "${AI_TRADING_RESEARCH_NOTIFY_SLACK:-1}" == "1" ]]; then
  if [[ "${AI_TRADING_RESEARCH_DRY_RUN:-0}" == "1" ]]; then
    echo "research automation ${CADENCE} dry-run completed; skipping Slack/OpenClaw completion notification" >&2
  elif [[ "${AI_TRADING_RESEARCH_PLAN_ONLY:-0}" != "1" || "${AI_TRADING_RESEARCH_NOTIFY_PLAN_ONLY:-0}" == "1" ]]; then
    set +e
    "${PYTHON_BIN}" -m ai_trading.tools.research_completion_notify \
      --cadence "${CADENCE}" \
      --workflow "${WORKFLOW:-${CADENCE}}" \
      --exit-code "${STATUS}" \
      --run-status "${RUN_STATUS}" \
      --channel "${AI_TRADING_RESEARCH_SLACK_CHANNEL:-#all-beatwallstreet}"
    NOTIFY_STATUS=$?
    set -e
    if [[ "${NOTIFY_STATUS}" != "0" ]]; then
      echo "research automation ${CADENCE} completion notification failed with exit ${NOTIFY_STATUS}; preserving run exit ${STATUS}" >&2
    fi
  fi
fi

exit "${STATUS}"
