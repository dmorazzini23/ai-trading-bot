# Slippage Spike Runbook

## Purpose

Respond when realized implementation shortfall spikes and threatens expectancy.

## Signals

- Elevated `is_bps` in TCA records
- Drop in fill quality (fill rate, latency, reject/cancel patterns)
- Frequent fallback/degraded data conditions

## Immediate Actions

1. Confirm data quality and provider health:

```bash
sudo journalctl -u ai-trading.service --since "30 min ago" --no-pager | \
grep -E 'BACKUP_PROVIDER_USED|EMPTY_DATA|PRIMARY_PROVIDER_SKIP_WINDOW_ACTIVE'
```

2. Review recent TCA slice:

```bash
tail -n 500 runtime/tca_records.jsonl
```

3. Reduce execution aggressiveness temporarily:

- Raise `AI_TRADING_ENTRY_COST_BUFFER_BPS`
- Lower `EXECUTION_MAX_NEW_ORDERS_PER_CYCLE`
- Increase passive execution preference knobs if enabled

4. Keep auto-promotion disabled during incident:

- `AI_TRADING_AFTER_HOURS_AUTO_PROMOTE=0`

## Recovery Actions

1. Rebuild rollups and cost model:

```bash
python -m ai_trading.tca.rollups
```

2. Re-run after-hours training smoke test and verify edge gates.
3. Re-enable normal pacing only after slippage normalizes over multiple cycles.

## Exit Criteria

- Slippage distribution returns near baseline
- No sustained fallback-provider degradation
- Expectancy/drawdown gates stabilize
