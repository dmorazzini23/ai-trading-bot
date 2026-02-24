#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
STRICT_MODE="${INSTITUTIONAL_STRICT:-1}"
SKIP_PYTEST="${INSTITUTIONAL_SKIP_PYTEST:-0}"
FAIL_ON_SKIP="${INSTITUTIONAL_FAIL_ON_SKIP:-${STRICT_MODE}}"

log() {
  printf '[institutional-gates] %s\n' "$*"
}

warn() {
  printf '[institutional-gates][warn] %s\n' "$*" >&2
}

fail() {
  printf '[institutional-gates][error] %s\n' "$*" >&2
  exit 1
}

run_acceptance_bundle() {
  local pytest_log
  pytest_log="$(mktemp)"
  local -a pytest_targets=(
    tests/test_runtime_contract_no_stubs.py
    tests/test_error_classification.py
    tests/test_dependency_breakers.py
    tests/test_retry_idempotent_reads_only.py
    tests/test_pretrade_price_collar.py
    tests/test_pretrade_max_order_size.py
    tests/test_pretrade_duplicate_block.py
    tests/test_pretrade_env_key_alignment.py
    tests/test_kill_switch_cancel_all.py
    tests/test_run_manifest_written.py
    tests/test_trading_mode_not_overwritten_by_cli.py
    tests/test_regime_profile_aggressive_present.py
    tests/test_liquidity_mode_balanced_supported.py
    tests/test_tca_implementation_shortfall.py
    tests/test_execution_report_daily_rollup.py
    tests/test_replay_engine_deterministic.py
    tests/test_walk_forward_no_leakage.py
    tests/test_order_types_supported_and_failfast.py
    tests/test_order_type_startup_failfast.py
    tests/test_portfolio_limits_vol_targeting.py
    tests/test_allocation_weight_updates.py
    tests/test_liquidity_participation_block.py
    tests/test_post_trade_learning_bounded_updates.py
    tests/test_quarantine_triggers_and_blocks.py
    tests/test_decision_record_config_snapshot.py
    tests/test_model_artifacts.py
    tests/test_model_verification_policy.py
    tests/integration/test_restart_reconciliation_exactly_once.py
    tests/unit/test_intent_store_database_url.py
    tests/unit/test_intent_store_idempotency.py
    tests/unit/test_intent_store_terminal_statuses.py
  )

  set +e
  pytest -q "${pytest_targets[@]}" | tee "${pytest_log}"
  local pytest_status="${PIPESTATUS[0]}"
  set -e

  if [[ "${pytest_status}" -ne 0 ]]; then
    rm -f "${pytest_log}"
    fail "Institutional acceptance bundle failed"
  fi

  if [[ "${FAIL_ON_SKIP}" = "1" ]]; then
    local skipped
    skipped="$("${PYTHON_BIN}" - "${pytest_log}" <<'PY'
import pathlib
import re
import sys

log_text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
matches = re.findall(r"(\d+)\s+skipped", log_text)
print(matches[-1] if matches else "0")
PY
)"
    if [[ "${skipped}" =~ ^[0-9]+$ ]] && (( skipped > 0 )); then
      rm -f "${pytest_log}"
      fail "Institutional acceptance bundle reported ${skipped} skipped tests (INSTITUTIONAL_FAIL_ON_SKIP=1)"
    fi
  fi

  rm -f "${pytest_log}"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    fail "Required file missing: $path"
  fi
}

check_env_keys() {
  local env_file="$1"
  local missing=0
  local keys=(
    AI_TRADING_ORDER_TYPES_ENABLED
    AI_TRADING_ORDER_TYPE_FAILFAST_ON_UNSUPPORTED
    AI_TRADING_PORTFOLIO_LIMITS_ENABLED
    AI_TRADING_LIQ_REGIME_ENABLED
    AI_TRADING_QUARANTINE_ENABLED
    AI_TRADING_TCA_ENABLED
  )
  for key in "${keys[@]}"; do
    if ! grep -Eq "^${key}=" "$env_file"; then
      warn "Missing env key in ${env_file}: ${key}"
      missing=1
    fi
  done
  if [[ "$missing" -ne 0 && "$STRICT_MODE" = "1" ]]; then
    fail "Required environment keys are missing from ${env_file}"
  fi
}

log "Checking repository artifacts"
require_file "docs/institutional_roadmap_30_60_90.md"
require_file "docs/acceptance_matrix.md"
require_file "ops/slo_thresholds.yaml"
require_file "runbooks/kill_switch_cancel_all.md"
require_file "runbooks/broker_outage_response.md"
require_file "runbooks/data_stale_fallback_response.md"
require_file "runbooks/deploy_rollback.md"

if [[ -f ".env" ]]; then
  log "Checking deployment env key coverage in .env"
  check_env_keys ".env"
fi

log "Running static checks"
log "Running secret exposure guard"
"${PYTHON_BIN}" tools/check_no_live_secrets.py

log "Running lint checks"
ruff check \
  ai_trading/core/runtime_contract.py \
  ai_trading/core/errors.py \
  ai_trading/core/dependency_breakers.py \
  ai_trading/core/retry.py \
  ai_trading/oms/pretrade.py \
  ai_trading/oms/orders.py \
  ai_trading/analytics/tca.py \
  ai_trading/replay/replay_engine.py \
  ai_trading/research/walk_forward.py \
  ai_trading/research/leakage_tests.py \
  ai_trading/models/artifacts.py \
  tests/test_runtime_contract_no_stubs.py \
  tests/test_error_classification.py \
  tests/test_dependency_breakers.py \
  tests/test_retry_idempotent_reads_only.py \
  tests/test_order_type_startup_failfast.py \
  tests/test_pretrade_env_key_alignment.py

log "Running mypy checks"
mypy \
  ai_trading/core/runtime_contract.py \
  ai_trading/core/errors.py \
  ai_trading/core/dependency_breakers.py \
  ai_trading/core/retry.py \
  ai_trading/oms/pretrade.py \
  ai_trading/oms/orders.py \
  ai_trading/analytics/tca.py \
  ai_trading/replay/replay_engine.py \
  ai_trading/research/walk_forward.py \
  ai_trading/research/leakage_tests.py \
  ai_trading/models/artifacts.py

if [[ "$SKIP_PYTEST" != "1" ]]; then
  log "Running institutional acceptance test bundle"
  run_acceptance_bundle
else
  warn "Skipping pytest bundle because INSTITUTIONAL_SKIP_PYTEST=1"
fi

log "Running bytecode compile check"
"${PYTHON_BIN}" -m py_compile $(git ls-files '*.py')

log "Institutional gates passed"
