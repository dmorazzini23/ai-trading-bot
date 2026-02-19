# Restart Reconciliation Runbook

## Purpose

Recover safely when the bot restarts while orders are in flight, using durable intent-state reconciliation.

## Preconditions

- `AI_TRADING_OMS_INTENT_STORE_ENABLED=1`
- `DATABASE_URL` configured for `EXECUTION_MODE=live` (Postgres recommended)
- `AI_TRADING_OMS_INTENT_STORE_PATH` may remain configured for sqlite fallback or migration source
- Service runs with broker access (`paper` or `live`)

## One-Time Migration (sqlite -> DATABASE_URL)

```bash
python3 scripts/migrate_oms_intent_store.py \
  --source-sqlite /home/aiuser/ai-trading-bot/runtime/oms_intents.db \
  --target-url "$DATABASE_URL"
```

Dry-run preview:

```bash
python3 scripts/migrate_oms_intent_store.py --dry-run
```

## What Happens Automatically

- On engine initialization, intents are reconciled with broker open orders.
- On each broker sync cycle, non-terminal intents are rechecked.
- Reconcile outcomes are logged as `OMS_INTENT_RECONCILE`.

## Manual Verification

1. Restart service:

```bash
sudo systemctl restart ai-trading.service
```

2. Check reconcile logs:

```bash
sudo journalctl -u ai-trading.service --since "10 min ago" --no-pager | grep OMS_INTENT_RECONCILE
```

3. Inspect store and ensure no stale `SUBMITTING`/`SUBMITTED` intents without matching broker orders.

## Expected Outcomes

- Intents with matching broker open orders remain active.
- Intents stuck without broker orders are marked `FAILED` with `reconcile_missing_broker_order`.
- No duplicate submissions for the same idempotency key.

## Escalation

- If repeated `marked_failed` spikes appear, pause new entries and inspect broker/API stability.
- If reconcile errors appear, verify DB file permissions and broker `list_orders` access.
