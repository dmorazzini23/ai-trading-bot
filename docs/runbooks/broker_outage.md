# Broker Outage Runbook

## Purpose

Protect capital when broker connectivity or order acknowledgements degrade.

## Detection

- `ALPACA_ORDER_SUBMIT_*` failures
- `BROKER_CONNECTIVITY` degraded/unknown for sustained window
- Reconciliation failures or repeated missing broker order states

## Immediate Actions

1. Confirm broker/API failures from logs:

```bash
sudo journalctl -u ai-trading.service --since "30 min ago" --no-pager | \
grep -E 'ALPACA_ORDER_SUBMIT|BROKER|RECON_MISMATCH|OMS_INTENT_RECONCILE'
```

2. Activate kill switch if outage is active:

- `AI_TRADING_KILL_SWITCH=1`
- Keep `AI_TRADING_CANCEL_ALL_ON_KILL=1`

3. Validate durable intent store status and DB connectivity:

```bash
python3 scripts/migrate_oms_intent_store.py --dry-run
```

## Recovery

1. Restore broker connectivity and verify account/orders endpoints.
2. Restart service and verify reconciliation logs:

```bash
sudo systemctl restart ai-trading.service
sudo journalctl -u ai-trading.service --since "10 min ago" --no-pager | grep OMS_INTENT_RECONCILE
```

3. Disable kill switch only after broker ACK path is stable.

## Exit Criteria

- Broker connectivity healthy for sustained interval.
- No reconcile drift and no duplicate submissions.
- Orders acknowledge/fill normally.
