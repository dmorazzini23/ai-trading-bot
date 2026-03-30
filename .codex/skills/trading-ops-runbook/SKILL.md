# Trading Ops Runbook Skill

Use this skill when you need fast operational triage for the live trading bot.

## Scope
- Validate runtime health and service state.
- Summarize go/no-go and execution KPIs.
- Inspect recent errors and provider degradation.
- Produce a concise action plan (immediate, next session, backlog).

## Workflow
1. Run `python scripts/ops_shift_check.py --phase auto` for a one-command shift summary.
2. Run `scripts/ops_runtime_check.sh` for full MCP catalog + diagnostics.
3. If health is degraded, run:
   - `python tools/mcp_ops_server.py --call recent_errors --args '{"unit":"ai-trading","since":"45 min ago"}'`
   - `python tools/mcp_observability_server.py --call journal_tail --args '{"unit":"ai-trading","since":"45 min ago","lines":250}'`
4. Pull runtime summary:
   - `python tools/mcp_runtime_data_server.py --call runtime_gonogo_status --args '{}'`
   - `python tools/mcp_observability_server.py --call runtime_kpi_snapshot --args '{}'`
5. If broker state is in question:
   - `python tools/mcp_broker_server.py --call runtime_broker_snapshot --args '{}'`
   - `python tools/mcp_broker_server.py --call alpaca_positions --args '{}'`
6. Respond with:
   - current state
   - top risks
   - concrete next commands

## Guardrails
- Do not place live orders from this skill.
- Keep restarts explicit and user-confirmed.
- Prefer read-only tools first; mutate ops state only when necessary.
