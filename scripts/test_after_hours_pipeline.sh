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
  tests/test_market_closed_gate.py \
  tests/test_market_closed_logging.py \
  tests/unit/test_market_closed_cycle.py \
  tests/test_main_execution_phase.py \
  tests/core/test_execution_engine_runtime.py \
  tests/execution/test_execution_runtime_controls.py \
  tests/execution/test_paper_bypass.py

echo "== after-hours training tests (serial) =="
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONHASHSEED=0 \
NO_XDIST=1 \
"${PYTHON_BIN}" tools/run_pytest.py -q \
  tests/test_after_hours_training.py

echo "== after-hours runtime canary =="
CANARY_LOG="$(mktemp -t ai-trading-after-hours-canary.XXXXXX.log)"
trap 'rm -f "${CANARY_LOG}"' EXIT

CANARY_ALPACA_API_KEY="${ALPACA_API_KEY:-DUMMYKEY}"
CANARY_ALPACA_SECRET_KEY="${ALPACA_SECRET_KEY:-DUMMYSECRET}"
CANARY_USING_DUMMY_CREDS=0
if [[ "${CANARY_ALPACA_API_KEY}" == DUMMY* || "${CANARY_ALPACA_SECRET_KEY}" == DUMMY* ]]; then
  CANARY_USING_DUMMY_CREDS=1
fi

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
ALPACA_API_KEY="${CANARY_ALPACA_API_KEY}" \
ALPACA_SECRET_KEY="${CANARY_ALPACA_SECRET_KEY}" \
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

EXPECTED_AUTH_ERR_RE='ALPACA_SDK_ACCOUNT_FAILED|ALPACA_AUTH_FAILED|AUTO_SIZING_ABORTED|POSITION_SIZING_FALLBACK|unauthorized|http_error:401'

if command -v rg >/dev/null 2>&1; then
  CANARY_ERR_LINES="$(rg -n '("level":\s*"ERROR"|"level":\s*"CRITICAL"| ERROR | CRITICAL )' "${CANARY_LOG}" || true)"
  if [[ -n "${CANARY_ERR_LINES}" ]]; then
    if [[ "${CANARY_USING_DUMMY_CREDS}" -eq 1 ]]; then
      CANARY_UNEXPECTED_ERR_LINES="$(printf '%s\n' "${CANARY_ERR_LINES}" | rg -v "${EXPECTED_AUTH_ERR_RE}" || true)"
      if [[ -n "${CANARY_UNEXPECTED_ERR_LINES}" ]]; then
        echo "after-hours canary logged unexpected ERROR/CRITICAL lines:"
        printf '%s\n' "${CANARY_UNEXPECTED_ERR_LINES}"
        exit 1
      fi
      echo "after-hours canary observed expected auth errors with dummy credentials"
    else
      echo "after-hours canary logged ERROR/CRITICAL"
      printf '%s\n' "${CANARY_ERR_LINES}"
      exit 1
    fi
  fi
  if ! rg -n 'SCHEDULER_COMPLETE' "${CANARY_LOG}" >/dev/null; then
    echo "after-hours canary did not reach scheduler completion"
    exit 1
  fi
else
  CANARY_ERR_LINES="$(grep -En '("level":[[:space:]]*"ERROR"|"level":[[:space:]]*"CRITICAL"| ERROR | CRITICAL )' "${CANARY_LOG}" || true)"
  if [[ -n "${CANARY_ERR_LINES}" ]]; then
    if [[ "${CANARY_USING_DUMMY_CREDS}" -eq 1 ]]; then
      CANARY_UNEXPECTED_ERR_LINES="$(printf '%s\n' "${CANARY_ERR_LINES}" | grep -Ev "${EXPECTED_AUTH_ERR_RE}" || true)"
      if [[ -n "${CANARY_UNEXPECTED_ERR_LINES}" ]]; then
        echo "after-hours canary logged unexpected ERROR/CRITICAL lines:"
        printf '%s\n' "${CANARY_UNEXPECTED_ERR_LINES}"
        exit 1
      fi
      echo "after-hours canary observed expected auth errors with dummy credentials"
    else
      echo "after-hours canary logged ERROR/CRITICAL"
      printf '%s\n' "${CANARY_ERR_LINES}"
      exit 1
    fi
  fi
  if ! grep -Eq 'SCHEDULER_COMPLETE' "${CANARY_LOG}"; then
    echo "after-hours canary did not reach scheduler completion"
    exit 1
  fi
fi

echo "after-hours pipeline checks passed"
