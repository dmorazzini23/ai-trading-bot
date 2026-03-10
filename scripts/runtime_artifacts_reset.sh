#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ENV_FILE_DATA_ROOT=""
if [[ -f "${ROOT_DIR}/.env" ]]; then
  ENV_FILE_DATA_ROOT="$(
    awk -F= '/^[[:space:]]*AI_TRADING_DATA_DIR[[:space:]]*=/{print $2}' "${ROOT_DIR}/.env" \
      | tail -n 1 \
      | sed -e 's/[[:space:]]#.*$//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
            -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//"
  )"
fi

DATA_ROOT="${AI_TRADING_DATA_DIR:-${STATE_DIRECTORY:-${ENV_FILE_DATA_ROOT}}}"
if [[ -n "${DATA_ROOT}" ]]; then
  DATA_ROOT="${DATA_ROOT%%:*}"
  if [[ "${DATA_ROOT}" != /* ]]; then
    DATA_ROOT="${ROOT_DIR}/${DATA_ROOT}"
  fi
else
  DATA_ROOT="${ROOT_DIR}"
fi

if [[ -n "${AI_TRADING_RUNTIME_DIR:-}" ]]; then
  RUNTIME_DIR="${AI_TRADING_RUNTIME_DIR}"
else
  RUNTIME_DIR="${DATA_ROOT}/runtime"
fi
if [[ -n "${AI_TRADING_RUNTIME_OWNER:-}" ]]; then
  RUNTIME_OWNER="${AI_TRADING_RUNTIME_OWNER}"
else
  RUNTIME_OWNER="$(stat -c '%U:%G' "${ROOT_DIR}" 2>/dev/null || true)"
  if [[ -z "${RUNTIME_OWNER}" || "${RUNTIME_OWNER}" == "root:root" ]]; then
    if id -u aiuser >/dev/null 2>&1; then
      RUNTIME_OWNER="aiuser:aiuser"
    else
      RUNTIME_OWNER="$(id -un):$(id -gn)"
    fi
  fi
fi
if [[ "${RUNTIME_OWNER}" != *:* ]]; then
  RUNTIME_OWNER="${RUNTIME_OWNER}:${RUNTIME_OWNER}"
fi
ARCHIVE_STAMP="$(date -u +%Y%m%d_%H%M%S)"
ARCHIVE_DIR="${RUNTIME_DIR}/archive/${ARCHIVE_STAMP}"
RUN_CYCLE=1

if [[ "${1:-}" == "--skip-cycle" ]]; then
  RUN_CYCLE=0
fi

resolve_path() {
  local raw="${1:-}"
  local fallback="${2}"
  local value="${raw:-${fallback}}"
  if [[ "${value}" = /* ]]; then
    printf '%s\n' "${value}"
  else
    printf '%s\n' "${DATA_ROOT}/${value}"
  fi
}

TRADE_HISTORY_PATH="$(resolve_path "${AI_TRADING_RUNTIME_PERF_TRADE_HISTORY_PATH:-}" "runtime/tca_records.jsonl")"
GATE_SUMMARY_PATH="$(resolve_path "${AI_TRADING_RUNTIME_PERF_GATE_SUMMARY_PATH:-}" "runtime/gate_effectiveness_summary.json")"

echo "[runtime-reset] data root: ${DATA_ROOT}"
echo "[runtime-reset] target runtime dir: ${RUNTIME_DIR}"
mkdir -p "${RUNTIME_DIR}" "${ARCHIVE_DIR}"

tracked_artifacts=(
  "after_hours_training_state.json"
  "coverage-ai_trading.json"
  "effective_policy.json"
  "gate_effectiveness.jsonl"
  "gate_effectiveness_summary.json"
  "rollout_state.json"
  "run_manifest.json"
  "run_manifest.jsonl"
)

echo "[runtime-reset] archiving tracked runtime artifacts -> ${ARCHIVE_DIR}"
for name in "${tracked_artifacts[@]}"; do
  source_path="${RUNTIME_DIR}/${name}"
  if [[ -f "${source_path}" ]]; then
    mv "${source_path}" "${ARCHIVE_DIR}/"
  fi
done

cat > "${RUNTIME_DIR}/after_hours_training_state.json" <<'JSON'
{
  "updated_at": "",
  "rows": 0,
  "dataset_fingerprint": "",
  "max_label_ts": "",
  "model_id": "",
  "model_name": ""
}
JSON

cat > "${RUNTIME_DIR}/coverage-ai_trading.json" <<'JSON'
{}
JSON

cat > "${RUNTIME_DIR}/effective_policy.json" <<'JSON'
{
  "ts": "",
  "loop_id": "",
  "policy_hash": "",
  "operational_safety_tier": "normal",
  "effective_policy": {}
}
JSON

cat > "${RUNTIME_DIR}/gate_effectiveness.jsonl" <<'JSONL'
{"ts":"","records_total":0,"accepted_records":0,"rejected_records":0,"excluded_records":0,"excluded_gate_counts":{},"gate_counts":{},"total_expected_net_edge_bps":0.0,"total_edge_proxy_bps":0.0,"gate_attribution":{},"symbol_attribution":{},"regime_attribution":{}}
JSONL

cat > "${RUNTIME_DIR}/gate_effectiveness_summary.json" <<'JSON'
{
  "total_records": 0,
  "total_accepted_records": 0,
  "total_rejected_records": 0,
  "excluded_records_total": 0,
  "gate_totals": {},
  "excluded_gate_totals": {},
  "total_expected_net_edge_bps": 0.0,
  "total_edge_proxy_bps": 0.0,
  "gate_effectiveness": {},
  "gate_attribution": {},
  "symbol_attribution": {},
  "regime_attribution": {},
  "updated_at": ""
}
JSON

cat > "${RUNTIME_DIR}/rollout_state.json" <<'JSON'
{
  "burn_in_paper_cycles": 0,
  "burn_in_paper_days": [],
  "burn_in_policy_hash": "",
  "burn_in_config_hash": "",
  "burn_in_reset_count": 0,
  "burn_in_last_reset_reason": "",
  "ramp_phase_index": 0,
  "ramp_phase_cycles": 0,
  "ramp_multiplier": 1.0,
  "ramp_last_transition": "",
  "updated_at": ""
}
JSON

cat > "${RUNTIME_DIR}/run_manifest.json" <<'JSON'
{
  "timestamp": "",
  "mode": "",
  "account_id": "",
  "git_commit_hash": "",
  "resolved_config_hash": "",
  "enabled_feature_flags": [],
  "runtime_contract": {},
  "effective_policy_hash": "",
  "effective_policy": {}
}
JSON

if [[ "${RUN_CYCLE}" == "1" ]]; then
  echo "[runtime-reset] running controlled paper cycle"
  # Run the single-cycle trade loop entrypoint directly so this command does
  # not require starting the API server.
  if ! IMPORT_PREFLIGHT_DISABLED=1 \
    INTERVAL_WHEN_CLOSED="${INTERVAL_WHEN_CLOSED:-1}" \
    AI_TRADING_INTERVAL="${AI_TRADING_INTERVAL:-1}" \
    AI_TRADING_ITERATIONS="${AI_TRADING_ITERATIONS:-1}" \
    "${PYTHON_BIN}" -c 'import sys; sys.argv=["ai_trading","--paper","--once","--interval","1"]; from ai_trading.__main__ import run_trade; run_trade()'; then
    echo "[runtime-reset] warning: controlled paper cycle failed; keeping clean baseline artifacts"
  fi
fi

echo "[runtime-reset] runtime performance snapshot"
"${PYTHON_BIN}" -m ai_trading.tools.runtime_performance_report \
  --trade-history "${TRADE_HISTORY_PATH}" \
  --gate-summary "${GATE_SUMMARY_PATH}" \
  --json \
  --go-no-go \
  --require-gate-valid || true

chown -R "${RUNTIME_OWNER}" "${RUNTIME_DIR}" 2>/dev/null || true
echo "[runtime-reset] done"
