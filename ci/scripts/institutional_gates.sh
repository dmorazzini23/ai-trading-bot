#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Prefer repo venv when available so pytest/ruff/mypy share one interpreter.
if [[ -z "${PYTHON_BIN:-}" && -x "./venv/bin/python" ]]; then
  PYTHON_BIN="./venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
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
    tests/unit/test_event_store_append_only.py
    tests/unit/test_oms_backtest_lifecycle_parity.py
    tests/unit/test_oms_lifecycle_parity_replay_tool.py
  )

  set +e
  "${PYTHON_BIN}" -m pytest -q "${pytest_targets[@]}" | tee "${pytest_log}"
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

run_lifecycle_parity_replay_smoke() {
  local fixture_path="tests/data/oms_lifecycle_parity_fixture.json"
  local tmp_db
  local tmp_out
  tmp_db="$(mktemp "${TMPDIR:-/tmp}/oms-lifecycle-parity.XXXXXX.db")"
  tmp_out="$(mktemp)"

  set +e
  "${PYTHON_BIN}" -m ai_trading.tools.oms_lifecycle_parity_replay \
    --fixture "${fixture_path}" \
    --database-url "sqlite:///${tmp_db}" \
    --intent-store-path "${tmp_db}" > "${tmp_out}"
  local replay_status="$?"
  set -e

  if [[ "${replay_status}" -ne 0 ]]; then
    rm -f "${tmp_db}" "${tmp_out}"
    fail "OMS lifecycle parity replay CLI smoke check failed"
  fi

  local replay_ok
  replay_ok="$("${PYTHON_BIN}" - "${tmp_out}" <<'PY'
import json
import pathlib
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
decoder = json.JSONDecoder()
payload = None
index = 0
while index < len(text):
    start = text.find("{", index)
    if start < 0:
        break
    try:
        candidate, end = decoder.raw_decode(text, start)
    except Exception:
        index = start + 1
        continue
    if isinstance(candidate, dict):
        payload = candidate
    index = end
print("1" if isinstance(payload, dict) and bool(payload.get("ok")) else "0")
PY
)"

  rm -f "${tmp_db}" "${tmp_out}"

  if [[ "${replay_ok}" != "1" ]]; then
    fail "OMS lifecycle parity replay reported mismatches"
  fi
}

run_phase2_execution_gate_check() {
  local require_phase2_gate
  require_phase2_gate="${AI_TRADING_INSTITUTIONAL_REQUIRE_PHASE2_GATE:-${INSTITUTIONAL_STRICT:-0}}"
  if [[ "${require_phase2_gate}" != "1" ]]; then
    log "Skipping Phase 2 execution-edge deploy gate (AI_TRADING_INSTITUTIONAL_REQUIRE_PHASE2_GATE!=1)"
    return
  fi
  local report_dir
  local max_age_hours
  report_dir="${AI_TRADING_EXECUTION_REPORT_DIR:-runtime/execution_reports}"
  max_age_hours="${AI_TRADING_INSTITUTIONAL_PHASE2_MAX_REPORT_AGE_HOURS:-36}"

  if ! "${PYTHON_BIN}" - "${report_dir}" "${max_age_hours}" <<'PY'
from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import sys


def _parse_ts(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


report_dir = Path(sys.argv[1]).expanduser()
max_age_hours = float(sys.argv[2])
if not report_dir.exists():
    print(
        f"[institutional-gates][error] Phase 2 gate report dir missing: {report_dir}",
        file=sys.stderr,
    )
    raise SystemExit(1)
matches = sorted(
    report_dir.glob("execution_report_*.json"),
    key=lambda path: path.stat().st_mtime,
)
if not matches:
    print(
        f"[institutional-gates][error] Phase 2 gate report missing in {report_dir}",
        file=sys.stderr,
    )
    raise SystemExit(1)
latest = matches[-1]
payload = json.loads(latest.read_text(encoding="utf-8"))
if not isinstance(payload, dict):
    print(
        f"[institutional-gates][error] Invalid execution report payload: {latest}",
        file=sys.stderr,
    )
    raise SystemExit(1)
roadmap = payload.get("roadmap")
phase2 = roadmap.get("phase_2_execution_edge") if isinstance(roadmap, dict) else None
if not isinstance(phase2, dict):
    print(
        f"[institutional-gates][error] Missing roadmap.phase_2_execution_edge in {latest}",
        file=sys.stderr,
    )
    raise SystemExit(1)
generated_at = _parse_ts(payload.get("generated_at"))
if generated_at is None:
    generated_at = datetime.fromtimestamp(latest.stat().st_mtime, tz=UTC)
age_hours = max(0.0, (datetime.now(UTC) - generated_at).total_seconds() / 3600.0)
if max_age_hours > 0 and age_hours > max_age_hours:
    print(
        (
            "[institutional-gates][error] "
            f"Execution report stale for phase 2 gate: age_hours={age_hours:.2f} "
            f"max_age_hours={max_age_hours:.2f} file={latest}"
        ),
        file=sys.stderr,
    )
    raise SystemExit(1)
if not bool(phase2.get("gate_passed", False)):
    print(
        (
            "[institutional-gates][error] "
            f"Phase 2 execution-edge gate failed in {latest}: "
            f"effective_gates={phase2.get('effective_gates')}"
        ),
        file=sys.stderr,
    )
    raise SystemExit(1)
print(
    (
        "[institutional-gates] Phase 2 execution-edge gate passed "
        f"(file={latest}, age_hours={age_hours:.2f})"
    )
)
PY
  then
    fail "Phase 2 execution-edge gate failed"
  fi
}

run_promotion_approval_gate_check() {
  local require_approval
  require_approval="${AI_TRADING_PROMOTION_REQUIRE_APPROVAL:-${INSTITUTIONAL_STRICT:-0}}"
  if [[ "${require_approval}" != "1" ]]; then
    log "Skipping promotion approval freshness gate (AI_TRADING_PROMOTION_REQUIRE_APPROVAL!=1)"
    return
  fi

  local governance_path
  local max_age_hours
  governance_path="${AI_TRADING_GOVERNANCE_BASE_PATH:-artifacts/governance}"
  max_age_hours="${AI_TRADING_INSTITUTIONAL_PROMOTION_APPROVAL_MAX_AGE_HOURS:-${AI_TRADING_PROMOTION_APPROVAL_MAX_AGE_HOURS:-168}}"

  if ! "${PYTHON_BIN}" -m ai_trading.tools.check_promotion_approval_gate \
    --governance-path "${governance_path}" \
    --max-age-hours "${max_age_hours}"; then
    fail "Promotion approval freshness gate failed"
  fi
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
require_file "runbooks/oms_lifecycle_parity_rollout.md"
require_file "tests/data/oms_lifecycle_parity_fixture.json"

if [[ -f ".env" ]]; then
  log "Checking deployment env key coverage in .env"
  check_env_keys ".env"
fi

log "Running static checks"
log "Running secret exposure guard"
"${PYTHON_BIN}" tools/check_no_live_secrets.py

log "Running lint checks"
"${PYTHON_BIN}" -m ruff check \
  ai_trading/core/runtime_contract.py \
  ai_trading/core/errors.py \
  ai_trading/core/dependency_breakers.py \
  ai_trading/core/retry.py \
  ai_trading/oms/pretrade.py \
  ai_trading/oms/orders.py \
  ai_trading/analytics/tca.py \
  ai_trading/oms/invariants.py \
  ai_trading/replay/replay_engine.py \
  ai_trading/tools/oms_lifecycle_parity_replay.py \
  ai_trading/tools/check_promotion_approval_gate.py \
  ai_trading/tools/update_phase2_execution_baseline.py \
  ai_trading/research/walk_forward.py \
  ai_trading/research/leakage_tests.py \
  ai_trading/models/artifacts.py \
  tests/test_runtime_contract_no_stubs.py \
  tests/test_error_classification.py \
  tests/test_dependency_breakers.py \
  tests/test_retry_idempotent_reads_only.py \
  tests/test_order_type_startup_failfast.py \
  tests/test_pretrade_env_key_alignment.py \
  tests/unit/test_event_store_append_only.py \
  tests/unit/test_oms_backtest_lifecycle_parity.py \
  tests/unit/test_oms_lifecycle_parity_replay_tool.py

log "Running mypy checks"
"${PYTHON_BIN}" -m mypy \
  ai_trading/core/runtime_contract.py \
  ai_trading/core/errors.py \
  ai_trading/core/dependency_breakers.py \
  ai_trading/core/retry.py \
  ai_trading/oms/pretrade.py \
  ai_trading/oms/orders.py \
  ai_trading/oms/invariants.py \
  ai_trading/analytics/tca.py \
  ai_trading/replay/replay_engine.py \
  ai_trading/tools/oms_lifecycle_parity_replay.py \
  ai_trading/tools/check_promotion_approval_gate.py \
  ai_trading/tools/update_phase2_execution_baseline.py \
  ai_trading/research/walk_forward.py \
  ai_trading/research/leakage_tests.py \
  ai_trading/models/artifacts.py

if [[ "$SKIP_PYTEST" != "1" ]]; then
  log "Running institutional acceptance test bundle"
  run_acceptance_bundle
  log "Running OMS lifecycle parity replay smoke check"
  run_lifecycle_parity_replay_smoke
else
  warn "Skipping pytest bundle because INSTITUTIONAL_SKIP_PYTEST=1"
fi

log "Running Phase 2 execution-edge gate check"
run_phase2_execution_gate_check

log "Running promotion approval freshness gate check"
run_promotion_approval_gate_check

log "Running bytecode compile check"
"${PYTHON_BIN}" -m py_compile $(git ls-files '*.py')

log "Institutional gates passed"
