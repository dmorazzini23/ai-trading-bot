#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-./venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

echo "== after-hours focused tests =="
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONHASHSEED=0 \
"${PYTHON_BIN}" tools/run_pytest.py -q \
  tests/test_run_cycle_after_hours.py \
  tests/test_after_hours_training.py \
  tests/test_market_closed_gate.py \
  tests/test_market_closed_logging.py \
  tests/unit/test_market_closed_cycle.py \
  tests/test_main_execution_phase.py \
  tests/core/test_execution_engine_runtime.py \
  tests/execution/test_execution_runtime_controls.py \
  tests/execution/test_paper_bypass.py

echo "== after-hours runtime canary =="
CANARY_LOG="$(mktemp -t ai-trading-after-hours-canary.XXXXXX.log)"
trap 'rm -f "${CANARY_LOG}"' EXIT

set +e
TESTING=1 \
ALLOW_AFTER_HOURS=1 \
AI_TRADING_AFTER_HOURS_TRAINING_ENABLED=0 \
AI_TRADING_AFTER_HOURS_RL_OVERLAY_ENABLED=0 \
AI_TRADING_HALT=1 \
INTERVAL_WHEN_CLOSED=1 \
API_PORT=19001 \
HEALTHCHECK_PORT=18081 \
RUN_HEALTHCHECK=1 \
EXECUTION_MODE=paper \
ALPACA_API_KEY="${ALPACA_API_KEY:-DUMMYKEY}" \
ALPACA_SECRET_KEY="${ALPACA_SECRET_KEY:-DUMMYSECRET}" \
ALPACA_TRADING_BASE_URL="${ALPACA_TRADING_BASE_URL:-https://paper-api.alpaca.markets}" \
ALPACA_DATA_FEED="${ALPACA_DATA_FEED:-iex}" \
TIMEFRAME="${TIMEFRAME:-1Min}" \
PYTHONUNBUFFERED=1 \
timeout "${AFTER_HOURS_CANARY_TIMEOUT:-180s}" \
"${PYTHON_BIN}" -m ai_trading.main --iterations "${AFTER_HOURS_CANARY_ITERATIONS:-1}" --interval "${AFTER_HOURS_CANARY_INTERVAL:-1}" \
>"${CANARY_LOG}" 2>&1
CANARY_RC=$?
set -e

cat "${CANARY_LOG}"
if [[ ${CANARY_RC} -ne 0 ]]; then
  echo "after-hours canary failed (exit=${CANARY_RC})"
  exit "${CANARY_RC}"
fi

if command -v rg >/dev/null 2>&1; then
  if rg -n '("level":\s*"ERROR"|"level":\s*"CRITICAL"| ERROR | CRITICAL )' "${CANARY_LOG}" >/dev/null; then
    echo "after-hours canary logged ERROR/CRITICAL"
    exit 1
  fi
  if ! rg -n 'SCHEDULER_COMPLETE' "${CANARY_LOG}" >/dev/null; then
    echo "after-hours canary did not reach scheduler completion"
    exit 1
  fi
else
  if grep -Eq '("level":[[:space:]]*"ERROR"|"level":[[:space:]]*"CRITICAL"| ERROR | CRITICAL )' "${CANARY_LOG}"; then
    echo "after-hours canary logged ERROR/CRITICAL"
    exit 1
  fi
  if ! grep -Eq 'SCHEDULER_COMPLETE' "${CANARY_LOG}"; then
    echo "after-hours canary did not reach scheduler completion"
    exit 1
  fi
fi

echo "after-hours pipeline checks passed"
