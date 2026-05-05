# Live Capital Runbook

This runbook is the operator path from paper trading to controlled live canary.
It does not grant live authority by itself.

## Pre-Market Checklist

1. Confirm the service is running:
   `systemctl status ai-trading.service --no-pager`
2. Confirm health:
   `curl -sS http://127.0.0.1:9001/healthz | jq .`
3. Confirm the active launch profile in health is expected.
4. Confirm broker, database, OMS invariants, replay governance, and provider
   authority are healthy.
5. Review the latest daily research answer:
   `jq . /var/lib/ai-trading-bot/runtime/research_reports/latest/daily_readiness_latest.json`
6. Review live-capital readiness:
   `jq . /var/lib/ai-trading-bot/runtime/live_capital_readiness_latest.json`
7. Confirm Slack/OpenClaw received the latest after-hours summary.

## Live Canary Requirements

Before `live_canary`, all must be true:

- Full validation is green and recorded as a validation artifact.
- `/healthz` is healthy.
- Broker is connected.
- Database and OMS gates are healthy.
- Replay governance is healthy.
- Promotion report is acceptable or explicitly waived in a documented manual
  review.
- Live cost model is ready with no unresolved cost breach.
- Provider is not degraded.
- Daily max loss is configured.
- `AI_TRADING_LAUNCH_PROFILE=live_canary`.
- Live account is explicitly confirmed only after account review.
- Canary symbols, max orders, max notional, quote-age cap, spread cap, and no
  shorts are intentional.

## Market-Open Checklist

1. Do not restart into live mode during a degraded provider state.
2. Confirm no stale halt flag is active.
3. Confirm `provider_authority.ok=true` in health.
4. Confirm `quotes_status.allowed=true` when quotes are available.
5. Confirm no open OMS violations.

## Intraday Monitoring

Watch:

- `/healthz` status and `attention_flags`
- Slack Go/No-Go alerts
- live cost breach alerts
- live canary event file:
  `/var/lib/ai-trading-bot/runtime/live_canary_events.jsonl`
- live canary state:
  `/var/lib/ai-trading-bot/runtime/live_canary_state_latest.json`

## Emergency Stop

If there is provider degradation, OMS violation, unexpected live order, cost
breach, or operator uncertainty:

1. Disable new entries through the existing runtime controls or launch profile.
2. Keep exits/reductions available.
3. Review broker state directly.
4. Capture `/healthz`, journal logs, live canary events, and order/fill
   artifacts.
5. Run the manual incident-replay workflow after the incident is stable.

## Daily Close

After market close:

```bash
scripts/run_research_automation.sh daily
jq . /var/lib/ai-trading-bot/runtime/research_reports/latest/daily_operator_summary.json
jq . /var/lib/ai-trading-bot/runtime/research_reports/latest/daily_readiness_latest.json
```

Use the generated trading-day report to review what the bot wanted, submitted,
rejected, filled, and missed.

## Escalation Rules

- `paper_trade` to `live_canary`: requires green readiness and manual approval.
- `live_canary` to `live_restricted`: requires multiple healthy sessions, no
  serious anomalies, acceptable costs, and positive evidence.
- `live_restricted` to `live_normal`: requires sustained evidence, reviewed
  drawdown profile, and explicit operator approval.

## Do Not Trade If

- Health is not healthy or ready.
- Provider authority is not OK.
- Broker is disconnected.
- Database or OMS gates are unhealthy.
- Replay governance is failed or stale beyond policy.
- Live cost model is missing or breached.
- Daily loss cap is missing.
- Launch profile is ambiguous.
- You cannot explain the latest daily readiness artifact.
