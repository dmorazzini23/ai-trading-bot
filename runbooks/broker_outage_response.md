# Runbook: Broker Outage / API Instability

## Trigger Conditions
- Elevated broker read/submit failures.
- Dependency breaker opens for `broker_submit`, `broker_positions`, or `broker_open_orders`.
- Auth failures (401/403) or persistent 5xx responses.

## Immediate Actions
1. Freeze trading (kill switch on).
2. Confirm breaker state and error category.
3. Validate whether issue is credentials, network, or broker-side outage.
4. Cancel open orders if risk posture requires a flat book.

## Commands
```bash
touch runtime/kill_switch
journalctl -u ai-trading.service -n 300 --no-pager | rg "broker_|AUTH_HALT|CIRCUIT_OPEN"
curl -sS http://127.0.0.1:8081/healthz
```

## Decision Tree
1. `AUTH_HALT`: rotate/repair credentials; do not resume until auth succeeds.
2. `RATE_LIMIT_RETRY`: reduce request rates and confirm recovery window clears.
3. `CIRCUIT_OPEN_broker_submit`: hold trading until breaker closes and dry-run checks pass.

## Recovery Criteria
1. Broker calls pass smoke checks for at least 10 continuous minutes.
2. Breakers closed for all broker dependencies.
3. Reconciliation status is clean.
4. Resume only in canary mode first.
