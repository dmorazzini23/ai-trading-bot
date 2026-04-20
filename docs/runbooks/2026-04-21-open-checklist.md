# 2026-04-21 Morning Open Checklist

## Goal

Resume the paper canary safely after the April 20 close-window fixes.

## Preconditions

- The April 20 close-window remediation is already deployed.
- Account ended flat after the prior session.
- [runtime/halt.flag](../../runtime/halt.flag) is still present overnight.

## Before Removing The Halt Flag

1. Confirm the service is up:

```bash
systemctl status ai-trading.service --no-pager
```

2. Confirm the account is flat and there are no open orders:

```bash
set -a
source .env.runtime >/dev/null 2>&1
./venv/bin/python - <<'PY'
from alpaca.trading.client import TradingClient
import os
client = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)
print({"positions": len(client.get_all_positions()), "open_orders": len(client.get_orders())})
PY
```

Expected:

- `positions: 0`
- `open_orders: 0`

3. Confirm runtime health is green:

```bash
curl -sS http://127.0.0.1:8081/healthz
```

Expected checks:

- `status=healthy`
- `database.backend=postgres`
- `oms_invariants.ok=true`
- `oms_lifecycle_parity.ok=true`
- `replay_live_parity_gate.ok=true`

4. Confirm the intended runtime scope from the live process:

Expected live values:

- `AI_TRADING_CANARY_SYMBOLS=AAPL,MSFT`
- `AI_TRADING_RUNTIME_SLEEVE_WHITELIST=day`
- `TRADING__ALLOW_SHORTS=0`
- `AI_TRADING_EOD_FLATTEN_ENABLED=1`
- `AI_TRADING_EOD_FLATTEN_LEAD_SECONDS=300`

## Unhalt

Remove the halt flag only after the checks above pass:

```bash
rm -f runtime/halt.flag
systemctl restart ai-trading.service
```

## First-Hour Monitoring

Watch for:

- first `AAPL` and `MSFT` entries
- first exits
- any short-open attempt
- any `ORDER_SKIPPED` or `ORDER_BLOCKED` spikes
- any replay/live parity or OMS lifecycle regressions

Useful log filter:

```bash
journalctl -u ai-trading.service -f | rg 'ALPACA_ORDER_|ORDER_|EOD_FLATTEN|BROKER_SYNC|OMS_|RUNTIME_GONOGO|REPLAY_LIVE'
```

## Midday Checks

- confirm open positions are only `AAPL` and/or `MSFT`
- confirm no short inventory exists
- confirm health remains green while orders fill

## Close-Window Checks

The critical behavior to verify on April 21:

- new opening orders must stop once the flatten lead window begins
- closing orders must still be allowed
- the account must be flat by or near the regular-session close

Specifically verify:

- `EOD_FLATTEN_TRIGGERED` appears
- no fresh opening `ALPACA_ORDER_SUBMIT_ATTEMPT` appears after flatten activation
- broker positions go to `0`
- open orders go to `0`

## After The Session

1. Compare today’s paper trades against:
   - [artifacts/offline_replay_recent.json](../../artifacts/offline_replay_recent.json)
2. Review:
   - entry/exit count
   - no-short compliance
   - flatten behavior
   - parity/OMS health
3. If any close-window regression reappears:
   - re-create `runtime/halt.flag`
   - restart the service
   - keep the account flat until reviewed
