# Data Staleness Runbook

## Purpose

Contain trading risk when minute/quote data is stale, sparse, or fallback-only.

## Detection

- `EMPTY_DATA`, `PRIMARY_PROVIDER_SKIP_WINDOW_ACTIVE`, `BACKUP_PROVIDER_USED`
- Quote staleness in `/healthz` (`quotes_status.allowed=false`, stale reason)

## Immediate Actions

1. Inspect recent provider diagnostics:

```bash
sudo journalctl -u ai-trading.service --since "30 min ago" --no-pager | \
grep -E 'EMPTY_DATA|BACKUP_PROVIDER_USED|PRIMARY_PROVIDER_SKIP_WINDOW_ACTIVE|DATA_SOURCE_FALLBACK'
```

2. Verify health endpoint state:

```bash
curl -sS http://127.0.0.1:${HEALTHCHECK_PORT:-8081}/healthz | jq .
```

3. Tighten degraded execution temporarily:

- Set `AI_TRADING_BLOCK_ENTRIES_ON_FALLBACK_MINUTE_DATA=1`
- Keep `EXECUTION_MARKET_ON_DEGRADED=0`
- Optionally set `AI_TRADING_KILL_SWITCH=1` if staleness is prolonged

## Recovery

1. Confirm primary feed recovery and quote freshness.
2. Ensure fallback usage drops to baseline.
3. Restart service after config updates:

```bash
sudo systemctl restart ai-trading.service
```

## Exit Criteria

- No repeated `EMPTY_DATA` bursts.
- `healthz` shows fresh quotes and normal provider state.
- Trading resumes without fallback-only dependency.
