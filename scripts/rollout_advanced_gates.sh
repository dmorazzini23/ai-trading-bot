#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/rollout_advanced_gates.sh <stage> [--env-file PATH] [--service NAME] [--restart] [--verify]

Stages:
  baseline    Keep advanced rollout features off (safe default)
  geometric   Enable geometric tiebreak only
  meta        Enable meta-label gate only
  bandit      Enable bandit routing only
  portfolio   Enable portfolio optimizer only
  status      Show current relevant env values

Examples:
  scripts/rollout_advanced_gates.sh baseline --restart --verify
  scripts/rollout_advanced_gates.sh geometric --restart --verify
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

STAGE="$1"
shift

ENV_FILE=".env.runtime"
SERVICE_NAME="ai-trading.service"
DO_RESTART=0
DO_VERIFY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="${2:-}"
      shift 2
      ;;
    --service)
      SERVICE_NAME="${2:-}"
      shift 2
      ;;
    --restart)
      DO_RESTART=1
      shift
      ;;
    --verify)
      DO_VERIFY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Env file not found: ${ENV_FILE}" >&2
  exit 1
fi

set_env_kv() {
  local key="$1"
  local value="$2"
  if grep -q "^${key}=" "${ENV_FILE}"; then
    sed -i "s|^${key}=.*|${key}=${value}|g" "${ENV_FILE}"
  else
    printf '%s=%s\n' "${key}" "${value}" >> "${ENV_FILE}"
  fi
}

apply_baseline_values() {
  # Keep advanced rollout controls disabled by default.
  set_env_kv "AI_TRADING_EXEC_BANDIT_ROUTING_ENABLED" "0"
  set_env_kv "AI_TRADING_EXEC_BANDIT_METHOD" "ucb"
  set_env_kv "AI_TRADING_EXEC_BANDIT_SCORE_WEIGHT" "0.25"
  set_env_kv "AI_TRADING_EXEC_BANDIT_EXPLORATION" "0.50"
  set_env_kv "AI_TRADING_EXEC_BANDIT_MIN_SAMPLES" "25"
  set_env_kv "AI_TRADING_EXEC_BANDIT_WINDOW_TRADES" "300"
  set_env_kv "AI_TRADING_EXEC_BANDIT_SESSION_BUCKET_ENABLED" "1"

  set_env_kv "AI_TRADING_EXEC_GEOMETRIC_TIEBREAK_ENABLED" "0"
  set_env_kv "AI_TRADING_EXEC_GEOMETRIC_TIEBREAK_WEIGHT" "0.08"
  set_env_kv "AI_TRADING_EXEC_GEOMETRIC_VARIANCE_PENALTY" "1.0"
  set_env_kv "AI_TRADING_EXEC_GEOMETRIC_DOWNSIDE_PENALTY" "0.75"
  set_env_kv "AI_TRADING_EXEC_GEOMETRIC_DRAWDOWN_PENALTY" "1.25"

  set_env_kv "AI_TRADING_EXECUTION_META_LABEL_GATE_ENABLED" "0"
  set_env_kv "AI_TRADING_EXECUTION_META_LABEL_REQUIRE_SCORE" "0"
  set_env_kv "AI_TRADING_EXECUTION_META_LABEL_MIN_SCORE" "0.52"

  set_env_kv "AI_TRADING_EXECUTION_EXPECTED_EDGE_CONFIDENCE_GATE_ENABLED" "0"
  set_env_kv "AI_TRADING_EXECUTION_EXPECTED_EDGE_CONFIDENCE_MIN_SAMPLES" "30"
  set_env_kv "AI_TRADING_EXECUTION_EXPECTED_EDGE_CONFIDENCE_LEVEL" "0.90"
  set_env_kv "AI_TRADING_EXECUTION_EXPECTED_EDGE_CONFIDENCE_MAX_RATIO" "2.2"
  set_env_kv "AI_TRADING_EXECUTION_EXPECTED_EDGE_CONFIDENCE_MARGIN_BPS" "0.5"

  set_env_kv "AI_TRADING_EXEC_PORTFOLIO_OPTIMIZER_ENABLED" "0"
  set_env_kv "AI_TRADING_EXEC_PORTFOLIO_OPTIMIZER_OPENINGS_ONLY" "1"
  set_env_kv "AI_TRADING_EXEC_PORTFOLIO_OPTIMIZER_IMPROVEMENT_THRESHOLD" "0.03"
  set_env_kv "AI_TRADING_EXEC_PORTFOLIO_OPTIMIZER_MAX_CORRELATION_PENALTY" "0.10"
  set_env_kv "AI_TRADING_EXEC_PORTFOLIO_OPTIMIZER_REBALANCE_DRIFT_THRESHOLD" "0.06"
  set_env_kv "AI_TRADING_EXEC_PORTFOLIO_OPTIMIZER_TURNOVER_PENALTY" "0.02"
  set_env_kv "AI_TRADING_EXEC_PORTFOLIO_OPTIMIZER_MODE" "execution_live"
}

show_status() {
  rg -n \
    "AI_TRADING_EXEC_(BANDIT_ROUTING_ENABLED|BANDIT_METHOD|BANDIT_SCORE_WEIGHT|BANDIT_EXPLORATION|BANDIT_MIN_SAMPLES|BANDIT_WINDOW_TRADES|GEOMETRIC_TIEBREAK_ENABLED|GEOMETRIC_TIEBREAK_WEIGHT|PORTFOLIO_OPTIMIZER_ENABLED|PORTFOLIO_OPTIMIZER_OPENINGS_ONLY|PORTFOLIO_OPTIMIZER_IMPROVEMENT_THRESHOLD|PORTFOLIO_OPTIMIZER_MAX_CORRELATION_PENALTY|PORTFOLIO_OPTIMIZER_REBALANCE_DRIFT_THRESHOLD|PORTFOLIO_OPTIMIZER_TURNOVER_PENALTY)|AI_TRADING_EXECUTION_(META_LABEL_GATE_ENABLED|META_LABEL_REQUIRE_SCORE|META_LABEL_MIN_SCORE|EXPECTED_EDGE_CONFIDENCE_GATE_ENABLED|EXPECTED_EDGE_CONFIDENCE_MIN_SAMPLES|EXPECTED_EDGE_CONFIDENCE_LEVEL|EXPECTED_EDGE_CONFIDENCE_MAX_RATIO|EXPECTED_EDGE_CONFIDENCE_MARGIN_BPS)" \
    "${ENV_FILE}" -S
}

case "${STAGE}" in
  baseline)
    apply_baseline_values
    ;;
  geometric)
    apply_baseline_values
    set_env_kv "AI_TRADING_EXEC_GEOMETRIC_TIEBREAK_ENABLED" "1"
    ;;
  meta)
    apply_baseline_values
    set_env_kv "AI_TRADING_EXECUTION_META_LABEL_GATE_ENABLED" "1"
    ;;
  bandit)
    apply_baseline_values
    set_env_kv "AI_TRADING_EXEC_BANDIT_ROUTING_ENABLED" "1"
    ;;
  portfolio)
    apply_baseline_values
    set_env_kv "AI_TRADING_EXEC_PORTFOLIO_OPTIMIZER_ENABLED" "1"
    ;;
  status)
    show_status
    exit 0
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    usage
    exit 2
    ;;
esac

echo "Applied stage '${STAGE}' to ${ENV_FILE}"
show_status

if [[ "${DO_RESTART}" == "1" ]]; then
  echo
  echo "Restarting ${SERVICE_NAME} ..."
  systemctl restart "${SERVICE_NAME}"
  systemctl --no-pager --full status "${SERVICE_NAME}" | sed -n '1,24p'
fi

if [[ "${DO_VERIFY}" == "1" ]]; then
  echo
  echo "Runtime report verification:"
  python3 - <<'PY'
import json
from pathlib import Path

report_path = Path("/var/lib/ai-trading-bot/runtime/runtime_performance_report_latest.json")
if not report_path.exists():
    print("report_missing=1")
    raise SystemExit(0)

payload = json.loads(report_path.read_text(encoding="utf-8"))
go = payload.get("go_no_go") or {}
obs = go.get("observed") or {}
gate = payload.get("gate_effectiveness") or {}
exec_vs_alpha = payload.get("execution_vs_alpha") or {}
print(f"go_no_go_gate_passed={go.get('gate_passed')}")
print(f"edge_realism_gap_ratio={exec_vs_alpha.get('edge_realism_gap_ratio')}")
print(f"execution_capture_ratio={exec_vs_alpha.get('execution_capture_ratio')}")
print(f"slippage_drag_bps={exec_vs_alpha.get('slippage_drag_bps')}")
print(f"acceptance_rate={gate.get('acceptance_rate')}")
print(f"top_rejection_concentration_ratio={gate.get('top_rejection_concentration_ratio')}")
print(f"top_rejection_concentration_gate={gate.get('top_rejection_concentration_gate')}")
print(f"expected_net_edge_bps={obs.get('expected_net_edge_bps')}")
print(f"expected_net_edge_bps_source={obs.get('expected_net_edge_bps_source')}")
PY
fi
