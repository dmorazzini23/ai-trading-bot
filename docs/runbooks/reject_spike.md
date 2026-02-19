# Reject Spike Runbook

## Purpose

Respond when order rejects increase beyond normal baseline and risk blocking execution.

## Detection

- `ORDER_REJECTED`, `ORDER_SUBMIT_FAILED`, or elevated reject counts in TCA/execution report.
- Increased `reject_rate` in daily execution report.

## Immediate Actions

1. Confirm broker/API health and current reject reasons:

```bash
sudo journalctl -u ai-trading.service --since "30 min ago" --no-pager | \
grep -E 'ORDER_REJECTED|ORDER_SUBMIT_FAILED|BROKER_CAPACITY|PDT|REJECT'
```

2. Check open/pending order pressure and pacing caps:

```bash
sudo journalctl -u ai-trading.service --since "30 min ago" --no-pager | \
grep -E 'ORDER_PACING_CAP_HIT|PENDING_NEW|CANCEL_ALL_TRIGGERED'
```

3. Reduce order pressure temporarily:

- Lower `EXECUTION_MAX_NEW_ORDERS_PER_CYCLE`.
- Raise `AI_TRADING_ENTRY_COST_BUFFER_BPS`.
- Keep `AI_TRADING_AFTER_HOURS_AUTO_PROMOTE=0`.

## Recovery

1. Verify reject reasons are resolved (permissions, order type, price collar, stale quotes).
2. Re-run a short paper smoke and inspect TCA records:

```bash
tail -n 200 runtime/tca_records.jsonl
```

3. Restore normal pacing only after reject rate normalizes over multiple cycles.

## Exit Criteria

- Reject rate returns to baseline.
- No sustained `ORDER_PACING_CAP_HIT` cascades.
- Fill quality and expectancy return to expected range.
