# Codex MCP / Skills Setup

This document wires the six operational features into your Codex/VS Code workflow.

## What Is Implemented In-Repo
1. `tools/mcp_runtime_data_server.py`
2. `tools/mcp_observability_server.py`
3. `tools/mcp_broker_server.py`
4. `tools/mcp_ops_server.py`
5. `tools/mcp_slack_alerts_server.py`
6. `tools/mcp_oncall_alerts_server.py`
7. `tools/mcp_metrics_query_server.py`
8. `tools/mcp_linear_issues_server.py`
9. `tools/mcp_secrets_manager_server.py`
10. `tools/mcp_sql_analytics_server.py`
11. `tools/mcp_market_events_server.py`
12. `tools/mcp_infra_cloud_server.py`
13. `.codex/skills/trading-ops-runbook/SKILL.md`
14. `.codex/skills/trading-ops-shift/SKILL.md`
15. `scripts/ops_shift_check.py` + `scripts/ops_runtime_check.sh`

## What Still Requires Manual Registration
- Registering MCP servers in your Codex client settings.
- Enabling GitHub/Gmail plugins in the Codex host account.
- Installing this repo-local skill into your Codex home if your client does not
  auto-discover `./.codex/skills`.

## MCP Server Registration (Template)
Create a client config entry (path depends on your Codex host):

```json
{
  "mcpServers": {
    "trading-runtime-data": {
      "command": "/home/aiuser/ai-trading-bot/venv/bin/python",
      "args": ["/home/aiuser/ai-trading-bot/tools/mcp_runtime_data_server.py"]
    },
    "trading-observability": {
      "command": "/home/aiuser/ai-trading-bot/venv/bin/python",
      "args": ["/home/aiuser/ai-trading-bot/tools/mcp_observability_server.py"]
    },
    "trading-broker": {
      "command": "/home/aiuser/ai-trading-bot/venv/bin/python",
      "args": ["/home/aiuser/ai-trading-bot/tools/mcp_broker_server.py"]
    },
    "trading-ops": {
      "command": "/home/aiuser/ai-trading-bot/venv/bin/python",
      "args": ["/home/aiuser/ai-trading-bot/tools/mcp_ops_server.py"]
    },
    "trading-slack-alerts": {
      "command": "/home/aiuser/ai-trading-bot/venv/bin/python",
      "args": ["/home/aiuser/ai-trading-bot/tools/mcp_slack_alerts_server.py"]
    },
    "trading-oncall-alerts": {
      "command": "/home/aiuser/ai-trading-bot/venv/bin/python",
      "args": ["/home/aiuser/ai-trading-bot/tools/mcp_oncall_alerts_server.py"]
    },
    "trading-metrics-query": {
      "command": "/home/aiuser/ai-trading-bot/venv/bin/python",
      "args": ["/home/aiuser/ai-trading-bot/tools/mcp_metrics_query_server.py"]
    },
    "trading-linear-issues": {
      "command": "/home/aiuser/ai-trading-bot/venv/bin/python",
      "args": ["/home/aiuser/ai-trading-bot/tools/mcp_linear_issues_server.py"]
    },
    "trading-secrets-manager": {
      "command": "/home/aiuser/ai-trading-bot/venv/bin/python",
      "args": ["/home/aiuser/ai-trading-bot/tools/mcp_secrets_manager_server.py"]
    },
    "trading-sql-analytics": {
      "command": "/home/aiuser/ai-trading-bot/venv/bin/python",
      "args": ["/home/aiuser/ai-trading-bot/tools/mcp_sql_analytics_server.py"]
    },
    "trading-market-events": {
      "command": "/home/aiuser/ai-trading-bot/venv/bin/python",
      "args": ["/home/aiuser/ai-trading-bot/tools/mcp_market_events_server.py"]
    },
    "trading-infra-cloud": {
      "command": "/home/aiuser/ai-trading-bot/venv/bin/python",
      "args": ["/home/aiuser/ai-trading-bot/tools/mcp_infra_cloud_server.py"]
    }
  }
}
```

## Connector Env Variables

### Slack incident connector
- `AI_TRADING_SLACK_WEBHOOK_URL`
- optional: `AI_TRADING_INCIDENT_MIN_CAPTURE_RATIO` (default `0.08`)
- optional dedupe-state path: `AI_TRADING_SLACK_INCIDENT_STATE_PATH`
- optional channel override: `AI_TRADING_SLACK_CHANNEL`
- optional enable/disable: `AI_TRADING_CONNECTOR_SLACK_ENABLED` (`1`/`0`)

Example tool call:

```bash
python tools/mcp_slack_alerts_server.py \
  --call notify_incident_channel \
  --args '{"on_change_only": true}'
```

### Slack end-of-day summary connector
- `AI_TRADING_SLACK_WEBHOOK_URL`
- optional channel override: `AI_TRADING_SLACK_EOD_CHANNEL`
- optional dedupe-state path: `AI_TRADING_SLACK_EOD_STATE_PATH`
- optional require market closed: `AI_TRADING_SLACK_EOD_REQUIRE_MARKET_CLOSED` (`1`/`0`, default `1`)
- optional enable/disable: `AI_TRADING_CONNECTOR_SLACK_EOD_ENABLED` (`1`/`0`)

Example tool call:

```bash
python tools/mcp_slack_alerts_server.py \
  --call notify_eod_summary \
  --args '{"require_market_closed": true}'
```

### PagerDuty / Opsgenie on-call escalation connector
- `AI_TRADING_CONNECTOR_ONCALL_ENABLED` (`1`/`0`, default `0`)
- `AI_TRADING_ONCALL_ON_CHANGE_ONLY` (`1`/`0`, default `1`)
- optional provider list: `AI_TRADING_ONCALL_PROVIDERS` (`pagerduty,opsgenie`)
- optional dedupe-state path: `AI_TRADING_ONCALL_STATE_PATH`
- PagerDuty:
  - `AI_TRADING_PAGERDUTY_ROUTING_KEY`
- Opsgenie:
  - `AI_TRADING_OPSGENIE_API_KEY`
  - optional `AI_TRADING_OPSGENIE_ALERT_URL` (default `https://api.opsgenie.com/v2/alerts`)
  - optional `AI_TRADING_OPSGENIE_TEAM`

Example tool call:

```bash
python tools/mcp_oncall_alerts_server.py \
  --call notify_oncall_incident \
  --args '{"providers":"pagerduty,opsgenie"}'
```

### Prometheus/Grafana metrics connector
- Prometheus direct: `AI_TRADING_PROMETHEUS_URL`
- or Grafana proxy:
  - `AI_TRADING_GRAFANA_URL`
  - `AI_TRADING_GRAFANA_PROMETHEUS_UID`
  - optional `AI_TRADING_GRAFANA_API_TOKEN`
- optional query defaults:
  - `AI_TRADING_PROMQL_SLIPPAGE_DRAG_BPS`
  - `AI_TRADING_PROMQL_EXECUTION_CAPTURE_RATIO`
  - `AI_TRADING_PROMQL_ORDER_REJECT_RATE_PCT`

Example tool call:

```bash
python tools/mcp_metrics_query_server.py \
  --call execution_trends_snapshot \
  --args '{"duration_minutes": 240, "step_s": 60}'
```

### Grafana-Native Metrics Forwarding (Recommended)
If you want strict MCP clients to query live Grafana Cloud time-series (instead of
runtime-report fallback), install the local Prometheus forwarder:

- `packaging/systemd/ai-trading-metrics-forwarder.service`
- `scripts/install_metrics_forwarder.sh`
- `scripts/check_metrics_forwarder.sh`

Install/enable:

```bash
cd /home/aiuser/ai-trading-bot
source venv/bin/activate
bash scripts/install_metrics_forwarder.sh
```

Validate:

```bash
bash scripts/check_metrics_forwarder.sh
```

Config inputs (auto-discovery supported):
- preferred explicit remote_write:
  - `AI_TRADING_PROM_REMOTE_WRITE_URL`
  - `AI_TRADING_PROM_REMOTE_WRITE_USERNAME`
  - `AI_TRADING_PROM_REMOTE_WRITE_PASSWORD`
  - optional rate-limit tuning:
    - `AI_TRADING_PROMETHEUS_SCRAPE_INTERVAL` (default `30s`)
    - `AI_TRADING_PROMETHEUS_METRIC_REGEX` (default `ai_trading_.*|bot_.*|orders_.*|alpaca_.*`)
    - `AI_TRADING_PROM_RW_QUEUE_CAPACITY` (default `25000`)
    - `AI_TRADING_PROM_RW_QUEUE_MAX_SHARDS` (default `4`)
    - `AI_TRADING_PROM_RW_QUEUE_MIN_SHARDS` (default `1`)
    - `AI_TRADING_PROM_RW_QUEUE_MAX_SAMPLES_PER_SEND` (default `5000`)
    - `AI_TRADING_PROM_RW_QUEUE_BATCH_SEND_DEADLINE` (default `20s`)
    - `AI_TRADING_PROM_RW_SEND_METADATA` (default `0`, recommended for Grafana Cloud 429 mitigation)
- or Grafana datasource discovery:
  - `AI_TRADING_GRAFANA_URL`
  - `AI_TRADING_GRAFANA_PROMETHEUS_UID`
  - `AI_TRADING_GRAFANA_API_TOKEN`

Important:
- `AI_TRADING_GRAFANA_API_TOKEN` is used to read datasource metadata from Grafana.
- It is typically not accepted by remote_write.
- Use a dedicated MetricsPublisher/remote_write token in
  `AI_TRADING_PROM_REMOTE_WRITE_PASSWORD`.
- If you observe `429 Too Many Requests` from Grafana Cloud remote_write,
  reduce scrape/queue pressure via the optional tuning variables above.

### Linear regression issue connector
- `AI_TRADING_LINEAR_API_KEY` (or `LINEAR_API_KEY`)
- `AI_TRADING_LINEAR_TEAM_ID` (or `LINEAR_TEAM_ID`)
- optional: `AI_TRADING_LINEAR_ENDPOINT` (default `https://api.linear.app/graphql`)
- optional: `AI_TRADING_LINEAR_LABEL_IDS` (comma-separated)
- optional dedupe-state path: `AI_TRADING_LINEAR_REGRESSION_STATE_PATH`
- optional issue priority: `AI_TRADING_LINEAR_PRIORITY`
- optional dry run mode: `AI_TRADING_CONNECTOR_LINEAR_DRY_RUN` (`1`/`0`)
- optional enable/disable: `AI_TRADING_CONNECTOR_LINEAR_ENABLED` (`1`/`0`)

Example dry run:

```bash
python tools/mcp_linear_issues_server.py \
  --call create_regression_issue \
  --args '{"dry_run": true}'
```

### Secrets manager MCP connector
- status/readiness:
  - `python tools/mcp_secrets_manager_server.py --call secrets_backend_status --args '{}'`
  - `python tools/mcp_secrets_manager_server.py --call aws_secret_inventory --args '{}'`
- runtime sync:
  - `python tools/mcp_secrets_manager_server.py --call sync_runtime_env --args '{}'`
- migration (explicit confirm required):
  - `python tools/mcp_secrets_manager_server.py --call migrate_local_env_to_aws --args '{"confirm":true,"merge_existing":true}'`

### Read-only SQL analytics MCP connector
- inspect table:
  - `python tools/mcp_sql_analytics_server.py --call warehouse_status --args '{}'`
- query trade/execution history:
  - `python tools/mcp_sql_analytics_server.py --call query_trade_history_sql --args '{"query":"SELECT symbol, AVG(slippage_bps_norm) FROM trade_history GROUP BY symbol"}'`
- built-in query examples:
  - `python tools/mcp_sql_analytics_server.py --call execution_trend_examples --args '{}'`

### Market events MCP connector
- optional feed endpoint:
  - `AI_TRADING_MARKET_EVENTS_JSON_URL`
- tools:
  - `market_sessions`
  - `fetch_events`
  - `market_risk_window`

### Cloud/infra MCP connector
- tools:
  - `host_summary`
  - `service_status`
  - `journal_errors`
  - `controlled_restart` (explicit `{"confirm": true}` required)
  - `restart_audit_tail`
  - `metadata_probe`

## Automated Incident Dispatch Timer

Included:
- `scripts/connector_incident_dispatch.py`
- `packaging/systemd/ai-trading-connectors.service`
- `packaging/systemd/ai-trading-connectors.timer`
- `scripts/install_connector_timer.sh`

Install/enable:

```bash
cd /home/aiuser/ai-trading-bot
source venv/bin/activate
bash scripts/install_connector_timer.sh
```

Validate:

```bash
systemctl status ai-trading-connectors.timer --no-pager -l
systemctl list-timers ai-trading-connectors.timer --all --no-pager
journalctl -u ai-trading-connectors.service -n 120 -o cat --no-pager
```

Optional dispatch controls:
- `AI_TRADING_CONNECTOR_SLACK_ON_CHANGE_ONLY=1`
- `AI_TRADING_CONNECTOR_SLACK_EOD_ENABLED=1`
- `AI_TRADING_SLACK_EOD_REQUIRE_MARKET_CLOSED=1`
- `AI_TRADING_CONNECTOR_SLACK_EOD_FORCE=0`
- `AI_TRADING_CONNECTOR_ONCALL_ENABLED=0`
- `AI_TRADING_ONCALL_ON_CHANGE_ONLY=1`
- `AI_TRADING_ONCALL_PROVIDERS=pagerduty,opsgenie`
- `AI_TRADING_CONNECTOR_FAIL_ON_ERROR=0` (default fail-open for timer stability)

## GitHub Workflow Feature
- Script: `scripts/github/pr_workflow.sh`
- Usage:

```bash
scripts/github/pr_workflow.sh <owner/repo> <pr-number>
```

This gives PR metadata, changed files, review comments, conversation comments,
and check status summary in one pass.

## Skill Installation (If Needed)
If your Codex host requires global skills:

```bash
mkdir -p "$HOME/.codex/skills/trading-ops-runbook"
cp -R /home/aiuser/ai-trading-bot/.codex/skills/trading-ops-runbook/* \
  "$HOME/.codex/skills/trading-ops-runbook/"

mkdir -p "$HOME/.codex/skills/trading-ops-shift"
cp -R /home/aiuser/ai-trading-bot/.codex/skills/trading-ops-shift/* \
  "$HOME/.codex/skills/trading-ops-shift/"
```

One-command shift check:

```bash
python scripts/ops_shift_check.py --phase auto | jq .
```

## Validation
Run:

```bash
cd /home/aiuser/ai-trading-bot
source venv/bin/activate
scripts/ops_runtime_check.sh
```

## Notes
- All broker endpoints in `mcp_broker_server.py` are read-only.
- `mcp_ops_server.py` requires explicit `{"confirm": true}` for restart.
- `mcp_slack_alerts_server.py` dedupes repeated alerts by fingerprint.
- `mcp_linear_issues_server.py` dedupes repeated issues by fingerprint.
- `mcp_oncall_alerts_server.py` adds PagerDuty/Opsgenie escalation with dedupe.
- `mcp_infra_cloud_server.py` writes restart actions to infra audit JSONL.
- Secrets-manager support is available in `scripts/runtime_env_sync.py`.
  See `docs/SECRETS_MANAGER_MIGRATION.md` to move secrets from `.env` into AWS
  Secrets Manager and render `.env.runtime` at sync time.
- These server scripts now support true MCP JSON-RPC stdio transport
  (Content-Length framing + `initialize`/`tools/list`/`tools/call`) for strict
  MCP clients.
- Backward-compatible CLI contract is still available for local scripts:
  - `--list-tools`
  - `--call <tool> --args '<json>'`
