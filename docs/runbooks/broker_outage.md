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

4. Verify failover policy toggles and defaults:

- `AI_TRADING_BROKER_FAILOVER_ENABLED` (default: `0`)
- `AI_TRADING_BROKER_FAILOVER_PROVIDER` (default: `paper`)
- `AI_TRADING_BROKER_FAILOVER_PROVIDERS` (optional ordered list, e.g. `paper,tradier`)
- `AI_TRADING_BROKER_FAILOVER_PROVIDER_COOLDOWN_SEC` (default: `120`)
- `AI_TRADING_BROKER_FAILOVER_POST_SUBMIT_RECONCILE_ENABLED` (default: `1`)

## Recovery

1. Restore broker connectivity and verify account/orders endpoints.
2. Restart service and verify reconciliation logs:

```bash
sudo systemctl restart ai-trading.service
sudo journalctl -u ai-trading.service --since "10 min ago" --no-pager | grep OMS_INTENT_RECONCILE
```

3. Disable kill switch only after broker ACK path is stable.

4. Inspect execution-quality diagnostics from control-plane:

```bash
curl -sS http://127.0.0.1:9001/operator/control-plane | jq '.snapshot.execution_quality'
```

Look for:
- `submit_reject_reasons_top`
- `cancel_reasons_top`
- `realized_slippage_decomposition`
- `event_outcomes_by_scope`

## Exit Criteria

- Broker connectivity healthy for sustained interval.
- No reconcile drift and no duplicate submissions.
- Orders acknowledge/fill normally.
