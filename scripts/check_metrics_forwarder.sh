#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_PORT="${AI_TRADING_PROMETHEUS_LOCAL_PORT:-19090}"

if [[ -f "${REPO_ROOT}/.env.runtime" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${REPO_ROOT}/.env.runtime"
  set +a
  LOCAL_PORT="${AI_TRADING_PROMETHEUS_LOCAL_PORT:-${LOCAL_PORT}}"
fi

echo "=== metrics forwarder service ==="
sudo systemctl status ai-trading-metrics-forwarder.service --no-pager -l | sed -n '1,120p'
ACTIVE_SINCE="$(systemctl show ai-trading-metrics-forwarder -p ActiveEnterTimestamp --value)"

echo "=== local prometheus readiness ==="
curl -fsS "http://127.0.0.1:${LOCAL_PORT}/-/ready"
echo

echo "=== local prometheus ai_trading metrics ==="
for metric in \
  ai_trading_slippage_drag_bps \
  ai_trading_execution_capture_ratio \
  ai_trading_order_reject_rate_pct
do
  echo "--- ${metric} ---"
  curl -fsS "http://127.0.0.1:${LOCAL_PORT}/api/v1/query" \
    --get \
    --data-urlencode "query=${metric}" | jq .
done

echo "=== remote_write transport health (local prometheus) ==="
curl -fsS "http://127.0.0.1:${LOCAL_PORT}/metrics" | \
  grep -E '^prometheus_remote_storage_(samples_total|samples_failed_total|samples_pending)' || true

echo "=== remote_write errors since current forwarder start ==="
sudo journalctl -u ai-trading-metrics-forwarder --since "${ACTIVE_SINCE}" -o cat | \
  grep -Ei '429|too many requests|401|invalid token|non-recoverable error|sending metadata' || \
  echo "none"

echo "=== MCP execution trends snapshot (grafana backend) ==="
cd "${REPO_ROOT}"
source venv/bin/activate
python tools/mcp_metrics_query_server.py \
  --call execution_trends_snapshot \
  --args '{"backend":"grafana","duration_minutes":240,"step_s":60}' \
  | jq -s 'map(select(has("tool"))) | last | {ok,result:{backend:.result.backend,fallback_used:.result.fallback_used,slippage:(.result.metrics.slippage_drag_bps.series[0].summary.latest // null),capture:(.result.metrics.execution_capture_ratio.series[0].summary.latest // null),reject:(.result.metrics.order_reject_rate_pct.series[0].summary.latest // null)}}'
