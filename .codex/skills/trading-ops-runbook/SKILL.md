# Trading Ops Runbook Skill

Use this skill when you need fast operational triage for the live trading bot.

## Scope
- Validate runtime health and service state.
- Summarize go/no-go and execution KPIs.
- Inspect recent errors and provider degradation.
- Produce a concise action plan (immediate, next session, backlog).

## Workflow
1. Run `scripts/ops_runtime_check.sh`.
2. If health is degraded, run:
   - `python tools/mcp_ops_server.py --call recent_errors --args '{"unit":"ai-trading","since":"45 min ago"}'`
   - `python tools/mcp_observability_server.py --call journal_tail --args '{"unit":"ai-trading","since":"45 min ago","lines":250}'`
3. Pull runtime summary:
   - `python tools/mcp_runtime_data_server.py --call runtime_gonogo_status --args '{}'`
   - `python tools/mcp_observability_server.py --call runtime_kpi_snapshot --args '{}'`
4. If broker state is in question:
   - `python tools/mcp_broker_server.py --call runtime_broker_snapshot --args '{}'`
   - `python tools/mcp_broker_server.py --call alpaca_positions --args '{}'`
5. Respond with:
   - current state
   - top risks
   - concrete next commands

## Guardrails
- Do not place live orders from this skill.
- Keep restarts explicit and user-confirmed.
- Prefer read-only tools first; mutate ops state only when necessary.

