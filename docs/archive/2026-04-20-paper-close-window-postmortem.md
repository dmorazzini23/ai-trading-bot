# 2026-04-20 Paper Close-Window Postmortem

## Scope

This postmortem covers the Monday, April 20, 2026 paper canary session after
the bot was reset into the intended `AAPL,MSFT` / no-shorts / EOD-flatten
configuration.

## Intended Runtime Profile

- Symbols: `AAPL,MSFT`
- `TRADING__ALLOW_SHORTS=0`
- `AI_TRADING_EOD_FLATTEN_ENABLED=1`
- `AI_TRADING_EOD_FLATTEN_LEAD_SECONDS=300`
- OMS backend: remote Postgres
- Replay reference:
  - [artifacts/offline_replay_recent.json](../../artifacts/offline_replay_recent.json)

Replay reference highlights:

- `symbols=2`
- `total_bars=814`
- `total_trades=13`
- `policy_controls_applied=true`
- `short_order_count=0`
- `short_fill_count=0`
- `violation_count=0`

## What Happened

### Session outcome

- First accepted canary long:
  - `MSFT` buy `2` at `2026-04-20T18:28:01Z`
  - `client_order_id=8777ad5cc2d0c512`
- No shorts were opened.
- The canary accumulated only long inventory during the session.
- The first flatten attempt inside the close window failed.
- Manual after-hours cleanup flattened the account successfully.
- Final account state after remediation:
  - `positions_count=0`
  - `open_orders_count=0`

### Broker-order summary after the canary reset

From the post-reset `AAPL/MSFT` paper orders:

- `MSFT`
  - submitted orders: `7`
  - filled orders: `4`
  - filled buy qty: `4`
  - filled sell qty: `4`
  - partial orders: `1`
  - canceled orders: `4`
- `AAPL`
  - submitted orders: `5`
  - filled orders: `4`
  - filled buy qty: `7`
  - filled sell qty: `7`
  - partial orders: `0`
  - canceled orders: `1`

Representative close-window broker events:

- `2026-04-20T19:55:38Z`:
  - `AAPL` buy `4` filled
- `2026-04-20T19:57:47Z`:
  - `EOD_FLATTEN_TRIGGERED`
- `2026-04-20T19:57:47Z`:
  - flatten attempted exits for `AAPL` and `MSFT`
  - both failed with:
    - `No shares available to exit for AAPL (requested 7, have 0)`
    - `No shares available to exit for MSFT (requested 4, have 0)`
- `2026-04-20T19:59:47Z`:
  - a new `MSFT` buy submit still occurred inside the flatten window
- `2026-04-20T20:02:31Z`:
  - manual market sell cleanup orders were accepted, then canceled after-hours
- `2026-04-20T20:05:14Z`:
  - manual after-hours limit sells filled:
    - `AAPL` sell `7`
    - `MSFT` sell `4`

## Root Causes

### 1. Flatten used stale local inventory instead of broker-truth positions

`exit_all_positions(...)` correctly passed broker-truth `raw_positions` into
`send_exit_order(...)`, but `send_exit_order(...)` still attempted its own
`ctx.api.get_position(symbol)` lookup and fell back to local zero inventory on
error. That caused EOD flatten to abort even while broker positions were still
open.

Fix landed in:

- [ai_trading/core/execution_flow.py](../../ai_trading/core/execution_flow.py)

### 2. Closing sells could be misclassified as short opens

Some exit paths did not consistently propagate `closing_position=True`, which
made long exits vulnerable to long-only short-sale guards.

Fix landed in:

- [ai_trading/execution/live_trading.py](../../ai_trading/execution/live_trading.py)

### 3. OMS lifecycle parity could degrade after filled intents

Reconcile logic could close intents without backfilling missing partial-fill
events, which caused lifecycle parity to fail during the active session.

Fix landed in:

- [ai_trading/execution/engine.py](../../ai_trading/execution/engine.py)

### 4. New openings were still allowed inside the flatten lead window

This was the remaining close-window policy gap. The bot could still submit new
opening orders after flatten mode was active.

Fix landed in:

- [ai_trading/execution/live_trading.py](../../ai_trading/execution/live_trading.py)

Opening orders are now blocked during the configured flatten lead window,
while closing orders remain allowed.

## Remediation Completed

- Fixed OMS reconcile/fill backfill to preserve lifecycle parity.
- Fixed sell-close inference so long exits stay reduce-only in long-only mode.
- Fixed EOD flatten to trust broker-truth positions when local inventory is stale.
- Fixed governance rollback-audit file ownership.
- Flattened the paper account after-hours and verified the account ended flat.
- Restarted the service in safe overnight mode with:
  - [runtime/halt.flag](../../runtime/halt.flag)
- Added a runtime submit gate that blocks new openings during the EOD flatten
  lead window.

## Comparison Against Offline Replay

The replay artifact and live session still agree on the most important
high-level constraints:

- canary universe only
- no short opens
- paper safety controls active

The main divergence was operational rather than directional:

- replay did not accumulate a late-session flatten failure
- live trading did
- replay also did not continue opening inventory after flatten mode activated
- live trading briefly did, before the new close-window gate was added

## Final State

At the end of remediation:

- paper account flat
- no open orders
- no short positions opened during the canary session
- service restarted successfully
- overnight trading blocked by halt flag pending morning operator review
