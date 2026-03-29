# Triage Checklist

## Pre-open
- `sync_env_runtime.sh` completed.
- Health: `ok=true` or `reason=market_closed`.
- `go_no_go.gate_passed=true`.
- No unresolved broker/local reconciliation drift.

## Intraday
- Provider status stable (`healthy` or expected temporary backup usage).
- No repeated `ERROR` / `Traceback` bursts in journald.
- Slippage drag and capture ratio trend in expected range.

## After-hours
- Refresh runtime reports.
- Confirm go/no-go payload parses correctly.
- Confirm model/training jobs ran as scheduled.
- Confirm artifacts persisted and no write fallbacks.

