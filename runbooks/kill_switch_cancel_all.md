# Runbook: Kill Switch + Cancel All

## Trigger Conditions
- Manual emergency halt.
- Automated halt reason in logs (`ALERT_HALT_TRADING`, `CIRCUIT_OPEN_*`, `AUTH_HALT`).
- Unexpected order behavior requiring immediate containment.

## Immediate Actions
1. Enable kill switch.
2. Confirm bot reports halted state.
3. Trigger cancel-all (if not already configured).
4. Verify open-order count reaches zero.

## Commands
```bash
# Halt via env/file toggle pattern used by this repo
touch runtime/kill_switch

# Inspect health and logs
curl -sS http://127.0.0.1:8081/healthz
journalctl -u ai-trading.service -n 200 --no-pager | rg "HALT|CANCEL_ALL|CIRCUIT_OPEN"
```

## Verification
1. Log contains `CANCEL_ALL_TRIGGERED`.
2. Reconciliation shows no lingering unexpected orders.
3. No new orders submitted while kill switch remains active.

## Recovery
1. Diagnose root cause before restart.
2. Remove kill switch file only after explicit approval.
3. Restart in canary-only mode (`AI_TRADING_CANARY_SYMBOLS=<small set>`).
