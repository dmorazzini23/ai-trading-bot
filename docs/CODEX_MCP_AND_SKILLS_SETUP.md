# Codex MCP / Skills Setup

This document covers the MCP servers and helper scripts that still exist in-repo.

## What Is Implemented In-Repo
1. `tools/mcp_runtime_data_server.py`
2. `tools/mcp_observability_server.py`
3. `tools/mcp_broker_server.py`
4. `tools/mcp_ops_server.py`
5. `tools/mcp_slack_alerts_server.py`
6. `tools/mcp_metrics_query_server.py`
7. `tools/mcp_secrets_manager_server.py`
8. `tools/mcp_sql_analytics_server.py`
9. `tools/mcp_market_events_server.py`
10. `tools/mcp_infra_cloud_server.py`
11. `.codex/skills/trading-ops-runbook/SKILL.md`
12. `.codex/skills/trading-ops-shift/SKILL.md`
13. `scripts/ops_shift_check.py` + `scripts/ops_runtime_check.sh`

## MCP Server Registration (Template)

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
    "trading-metrics-query": {
      "command": "/home/aiuser/ai-trading-bot/venv/bin/python",
      "args": ["/home/aiuser/ai-trading-bot/tools/mcp_metrics_query_server.py"]
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

### OpenClaw runtime incident connector
OpenClaw delivery is handled by `scripts/connector_incident_dispatch.py`, which posts to the OpenClaw connector dispatcher/gateway. It is not delivered by `mcp_slack_alerts_server.py`; that MCP server remains the Slack alert helper.

- optional enable/disable: `AI_TRADING_CONNECTOR_OPENCLAW_ENABLED` (`1`/`0`)
- optional change-only dedupe: `AI_TRADING_CONNECTOR_OPENCLAW_ON_CHANGE_ONLY` (`1`/`0`)
- optional webhook target override: `AI_TRADING_OPENCLAW_RUNTIME_WEBHOOK_URL`
- optional gateway URL override: `AI_TRADING_OPENCLAW_GATEWAY_URL`
- optional token override: `AI_TRADING_OPENCLAW_HOOK_TOKEN`
- optional config path override: `AI_TRADING_OPENCLAW_CONFIG_PATH`
- optional dedupe-state path: `AI_TRADING_OPENCLAW_INCIDENT_STATE_PATH`

## Automated Incident Dispatch Timer

Included:
- `scripts/connector_incident_dispatch.py`
- `packaging/systemd/ai-trading-connectors.service`
- `packaging/systemd/ai-trading-connectors.timer`
- `scripts/install_connector_timer.sh`

Optional dispatch controls:
- `AI_TRADING_CONNECTOR_SLACK_ON_CHANGE_ONLY=1`
- `AI_TRADING_CONNECTOR_SLACK_EOD_ENABLED=1`
- `AI_TRADING_SLACK_EOD_REQUIRE_MARKET_CLOSED=1`
- `AI_TRADING_CONNECTOR_SLACK_EOD_FORCE=0`
- `AI_TRADING_CONNECTOR_OPENCLAW_ENABLED=1`
- `AI_TRADING_CONNECTOR_OPENCLAW_ON_CHANGE_ONLY=1`
- `AI_TRADING_CONNECTOR_FAIL_ON_ERROR=0`

## Operator Assistant Policy

OpenClaw/Slack is the fast operator layer. It should answer from artifacts and
local health checks by default, not by running heavy work. Broad validation,
training, replay/backtests, and code patches should be handed off as Codex
`/goal` prompts unless the operator explicitly requests a deep implementation
session. Runtime incident payloads include an `operatorAssistantPolicy` object
with this boundary so OpenClaw can keep normal Slack chat quick while preserving
Codex for thorough code changes.

See [docs/OPERATOR_ASSISTANT_POLICY.md](/home/aiuser/ai-trading-bot/docs/OPERATOR_ASSISTANT_POLICY.md).

## Validation

```bash
cd /home/aiuser/ai-trading-bot
source venv/bin/activate
scripts/ops_runtime_check.sh
```

## Notes
- `mcp_slack_alerts_server.py` dedupes repeated alerts by fingerprint.
- OpenClaw incident and model-readiness notices are dispatched by the connector timer through the OpenClaw gateway/dispatcher, not by `mcp_slack_alerts_server.py`.
- `mcp_infra_cloud_server.py` writes restart actions to infra audit JSONL.
- Secrets-manager support is available in `scripts/runtime_env_sync.py`.
