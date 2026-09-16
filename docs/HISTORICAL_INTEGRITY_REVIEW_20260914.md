# Historical integrity and active reconciliation — September 14, 2026

Read-only review completed at 14:29 UTC. No historical repairs are indicated by
the checks below. No source/runtime changes, orders, restarts or notifications.

## Historical records and repair preview

Runtime configuration points both DATABASE_URL and AI_TRADING_OMS_INTENT_STORE_PATH
at `/var/lib/ai-trading-bot/runtime/oms_intents_paper_monday.db`. Inspected a
consistent read-only SQLite transaction: 2,082 intents, 715 fill rows, and 12,381
intent_store-source lifecycle events. Earliest intent creation is May 4.

Fresh paper activity capture begins May 3 and completed pagination at 14:28 UTC.
All 711 broker-linked stored fill quantities match broker activity sums by order,
symbol and side. Four unmatched fills have explicit metadata source
`live_cutover_drill` and synthetic broker IDs. They are simulated records, not
missing broker executions. Each filled intent has one persisted fill row.

Checks found zero quantity differences, fills above intended quantity, invalid
quantities, orphan fills, repeated same-value/timestamp fill groups, or duplicate
broker IDs across intents. No intent_store-source submit/fill events were recorded
after INTENT_CLOSED, and no currently open intent had a recorded prior closure.
SQLite quick_check returned ok. Backfilled lifecycle events were excluded from the
ordering check because ingestion order does not establish original event order.

Repair preview: **zero proposed database mutations**. Preserve all rows, including
the four identifiable drills; exclude those drills when interpreting broker-backed
execution evidence. This audit does not certify that every downstream consumer
filters simulated records. No delete/update is justified by these findings.

The initial 90-day capture omitted older fills. A broader March capture reached
its 100-page safety limit and was rejected as incomplete. The final bounded May
capture covers the full stored intent date range and completed pagination. No
claim relies on the incomplete intermediate result.

## Active reconciliation

The deployed service logs establish the active path:
LiveTradingEngine.synchronize_broker_state calls _reconcile_durable_intents,
delegating to OrderManager.reconcile_open_intents; it also calls
_reconcile_pending_order_runtime_artifacts. Logs at 14:12 show the durable intent
scan (zero errors), terminal pending-order reconciliation (one applied, zero
lookup failures), and fresh broker sync. Subsequent sync logs remain fresh.

The separate execution/position_reconciler.py periodic worker has no repository
production-startup call site. Its start method is only called by its exported
start_position_monitoring wrapper. Deployment scripts do not start that wrapper.
No POS_RECONCILE activation log was observed. Thus the verified live reconciliation
uses LiveTradingEngine, not that optional worker. This is source/log evidence;
thread names were generic Python names and cannot independently prove module
activation. No optional worker was enabled during this review.

The session is underway. An AAPL buy and subsequent sell entered the persisted
ledger during this audit; their quantities are included in the final comparison.
At 14:29 UTC broker state is fresh/connected with zero positions and orders.
Health status is degraded with the existing replay_live_parity_gate_failed and
required_model_stale flags. This does not establish strategy eligibility or
per-fill costs. Post-close session reconciliation remains pending.

## Evidence and limits

- Audit fixture: artifacts/audits/history_integrity_20260914.py; execute with
  ./venv/bin/python after refreshing its explicit /tmp broker input.
- Private audit results: /tmp/history-integrity-audit.json.
- Private broker capture/report: /tmp/history-audit-activities.json and
  /tmp/history-audit-accounting.json. Final capture uses --lookback-days 134.
- Health: /tmp/history-audit-final-health.json.
- Checks: read-only transaction, completed broker pagination, exact Decimal
  quantity comparison (1e-6 tolerance), SQLite quick_check, source/log tracing.
- Documentation-only validation and git diff --check passed. No runtime code
  changed, so runtime regression tests or deployment were not required.

This rules out the examined persisted corruption patterns in this snapshot. It
does not prove absence of transient past states, absent event writes, all possible
duplicates, causal quote lineage, or complete per-fill fees. These remain distinct
from quantity agreement. Holdout and research budgets are unchanged.
