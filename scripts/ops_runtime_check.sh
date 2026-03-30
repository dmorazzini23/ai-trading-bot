#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

echo "=== MCP tool catalogs ==="
"${PYTHON_BIN}" tools/mcp_runtime_data_server.py --list-tools | jq .
"${PYTHON_BIN}" tools/mcp_observability_server.py --list-tools | jq .
"${PYTHON_BIN}" tools/mcp_broker_server.py --list-tools | jq .
"${PYTHON_BIN}" tools/mcp_ops_server.py --list-tools | jq .
"${PYTHON_BIN}" tools/mcp_slack_alerts_server.py --list-tools | jq .
"${PYTHON_BIN}" tools/mcp_oncall_alerts_server.py --list-tools | jq .
"${PYTHON_BIN}" tools/mcp_metrics_query_server.py --list-tools | jq .
"${PYTHON_BIN}" tools/mcp_linear_issues_server.py --list-tools | jq .
"${PYTHON_BIN}" tools/mcp_secrets_manager_server.py --list-tools | jq .
"${PYTHON_BIN}" tools/mcp_sql_analytics_server.py --list-tools | jq .
"${PYTHON_BIN}" tools/mcp_market_events_server.py --list-tools | jq .
"${PYTHON_BIN}" tools/mcp_infra_cloud_server.py --list-tools | jq .

echo "=== Runtime KPI snapshot ==="
"${PYTHON_BIN}" tools/mcp_observability_server.py \
  --call runtime_kpi_snapshot \
  --args '{}' | jq .

echo "=== Runtime go/no-go ==="
"${PYTHON_BIN}" tools/mcp_runtime_data_server.py \
  --call runtime_gonogo_status \
  --args '{}' | jq .

echo "=== Health probe ==="
"${PYTHON_BIN}" tools/mcp_ops_server.py \
  --call health_probe \
  --args '{"port":8081}' | jq .

echo "=== Service status ==="
"${PYTHON_BIN}" tools/mcp_ops_server.py \
  --call service_status \
  --args '{"unit":"ai-trading"}' | jq .

echo "=== Slack incident snapshot ==="
"${PYTHON_BIN}" tools/mcp_slack_alerts_server.py \
  --call runtime_incident_snapshot \
  --args '{}' | jq .

echo "=== Metrics backend status ==="
"${PYTHON_BIN}" tools/mcp_metrics_query_server.py \
  --call metrics_backend_status \
  --args '{}' | jq .

echo "=== Linear regression snapshot ==="
"${PYTHON_BIN}" tools/mcp_linear_issues_server.py \
  --call runtime_regression_snapshot \
  --args '{}' | jq .

echo "=== On-call incident snapshot ==="
"${PYTHON_BIN}" tools/mcp_oncall_alerts_server.py \
  --call runtime_incident_snapshot \
  --args '{}' | jq .

echo "=== Secrets backend status ==="
"${PYTHON_BIN}" tools/mcp_secrets_manager_server.py \
  --call secrets_backend_status \
  --args '{}' | jq .

echo "=== SQL analytics examples ==="
"${PYTHON_BIN}" tools/mcp_sql_analytics_server.py \
  --call execution_trend_examples \
  --args '{}' | jq .

echo "=== Market events source status ==="
"${PYTHON_BIN}" tools/mcp_market_events_server.py \
  --call market_events_source_status \
  --args '{}' | jq .

echo "=== Infra host summary ==="
"${PYTHON_BIN}" tools/mcp_infra_cloud_server.py \
  --call host_summary \
  --args '{}' | jq .

echo "OPS_RUNTIME_CHECK_OK"
