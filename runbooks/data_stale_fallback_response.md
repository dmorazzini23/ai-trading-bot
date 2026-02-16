# Runbook: Stale Data / Fallback Degradation

## Trigger Conditions
- Data freshness SLO breach.
- Frequent fallback-provider usage.
- Breaker open on `data_primary` or `quotes_primary`.
- Repeated `BAD_DATA_CONTRACT` gates.

## Immediate Actions
1. Confirm stale-data scope (symbol-specific vs global).
2. Switch to containment mode:
   - no new entries
   - optionally keep only risk-reduction exits
3. Validate fallback data quality and quote age controls.

## Commands
```bash
journalctl -u ai-trading.service -n 300 --no-pager | rg "BAD_DATA|fallback|CIRCUIT_OPEN_data|MARKET_CLOSED_BLOCK"
curl -sS http://127.0.0.1:8081/healthz
```

## Tuning Levers
1. Tighten entry gating when fallback is active.
2. Increase cost buffer and reduce participation caps for thin liquidity.
3. Restrict to canary symbols while primary data recovers.

## Recovery Criteria
1. Primary feed healthy for required consecutive passes.
2. No stale-data alerts for one full session interval.
3. Replay/decision logs show normal gating distribution.
