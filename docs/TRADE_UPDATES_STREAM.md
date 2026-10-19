# Alpaca trade-updates stream (shadow mode)

The broker pushes order lifecycle events over the Alpaca trade-updates
websocket. The runtime subscribes in **shadow mode**: every pushed update is
recorded to its own artifact and compared with the polled fill record. It is
not authoritative. It never writes the OMS event store or
`runtime/fill_events.jsonl`, and it does not feed any trading decision or gate.
Polling (`poll_order_fill_status` and the execution engine's reconciliation)
is still the only source of fills.

## Runtime

- Entry point: `ai_trading.alpaca_api.start_trade_updates_stream` (alpaca-py
  0.42.1 `TradingStream`, `raw_data=True`). `python -m ai_trading` starts it
  in the daemon thread `trade-updates-stream` before the main loop (skipped in
  test mode). The legacy `bot_engine.main` path starts it through
  `startup_runtime._run_trade_updates_stream`.
- Paper or live comes from process config: `ALPACA_TRADING_BASE_URL`
  containing `paper` means paper. With no base URL, `EXECUTION_MODE=live`
  means live, and anything else means paper.
- Kill switch: `AI_TRADING_TRADE_UPDATES_STREAM_ENABLED=0` (default `1`).
- Reconnect: capped exponential backoff with jitter (50–100% of
  `min(cap, base * 2^n)`). `AI_TRADING_TRADE_UPDATES_STREAM_BACKOFF_BASE_SEC`
  defaults to 1 and `..._BACKOFF_MAX_SEC` to 60. The step resets after a
  connection that lasted 30 s or longer.
- Logs: `TRADE_UPDATES_STREAM_CONNECTED`, `_DISCONNECTED` (with `reason`),
  `_RECONNECTING`, `_EVENT`, `_DISABLED`, `_STOPPED`, `_WRITE_FAILED` (at most
  once per minute).
- Errors are logged and never leave the stream thread. A shadow-stream failure
  never changes service status.

## Artifact: `runtime/trade_updates_stream.jsonl`

Override the path with `AI_TRADING_TRADE_UPDATES_STREAM_PATH`. It resolves
like other runtime artifacts (`AI_TRADING_DATA_DIR`, then `STATE_DIRECTORY`).
Each update is one JSON line, written with a single `O_APPEND` write:

| field | source |
| --- | --- |
| `received_at` | local UTC receive time |
| `event` | `new`, `partial_fill`, `fill`, `canceled`, `expired`, `rejected`, `replaced`, `pending_*`, ... |
| `event_timestamp` | broker event `timestamp` |
| `order_timestamps` | the order's `created_at`, `updated_at`, `submitted_at`, `filled_at`, `canceled_at`, `expired_at`, `failed_at`, `replaced_at` (when present) |
| `order_id`, `client_order_id`, `symbol`, `side`, `order_status` | order |
| `order_qty` | order `qty` |
| `qty`, `price` | the execution's quantity and price (fill events) |
| `filled_qty`, `filled_avg_price` | cumulative, from the order |
| `position_qty` | position after the execution |
| `execution_id` | when present |
| `raw` | the full broker message |

De-duplication: the key is (`order_id`, `event`, `event_timestamp`,
`execution_id`, or `filled_qty` when there is no execution ID), held in a
bounded in-memory set of 20,000 keys. A replay after a reconnect is not
recorded twice. Retention: 32 MiB / 100,000 lines through
`scripts/prune_runtime_jsonl.sh` (archive-only).

## Parity

```bash
python -m ai_trading.tools.trade_updates_parity --session 2026-10-16
```

The tool reads the shadow artifact and `fill_events.jsonl` read-only for the
New York session date. It writes JSON to stdout (and to `--output PATH` when
given) and a short summary to stderr. Stream and polled fills are matched per `order_id` on cumulative filled
quantity. The stream side uses the highest `filled_qty` among `fill` and
`partial_fill` events. The polled side sums `fill_qty` over unique `fill_id`
values. The report includes:

- `stream_fill_count` and `polled_fill_count`, with order counts
- `matched`, `stream_only`, `polled_only` and `qty_mismatch`
- `price_delta_bps_median` and `price_delta_bps_p95`: stream
  `filled_avg_price` against the polled qty-weighted fill price
- `latency_seconds_median` and `latency_seconds_p95`: from the broker fill
  timestamp to the latest polled `ts`
- `mismatches`, capped at 200
- `available` and `reason`: `stream_artifact_missing`,
  `no_stream_events_for_session`, `polled_fill_record_missing` or `ok`
- `clean` (see below)

The stream thread refreshes `runtime/trade_updates_parity_latest.json` every
`AI_TRADING_TRADE_UPDATES_PARITY_INTERVAL_SEC` (default 300; `0` disables).
It skips the refresh when neither input file has changed. The work runs off
the event loop and never on a request path.

## Health

`/healthz` carries a `trade_updates_stream` section with `enabled`,
`connected`, `degraded`, `shadow_only: true`, `paper`, `last_event_at`,
`events_today` (New York date), `reconnects`, `duplicates_skipped`,
`write_failures` and `parity` (the latest summary fields, without the mismatch
list). It is built from in-memory state only.

The section sets `degraded` when the stream is enabled but has not connected
within `AI_TRADING_TRADE_UPDATES_STREAM_GRACE_SEC` (default 120 s). A degraded
section adds the existing `trade_updates_stream_degraded` attention flag. The
flag is informational: it does not change `ok`, readiness or any gate.

## Criteria for making the stream authoritative

A separate, reviewed change may make the stream a source of fills only after
all of the following hold:

1. Three consecutive regular sessions with at least one stream fill each, and
   for each session `stream_only == 0`, `polled_only == 0` and
   `qty_mismatch == 0` (`clean: true`).
2. `price_delta_bps_p95 < 1.0` in each of those sessions.
3. No `TRADE_UPDATES_STREAM_WRITE_FAILED`, and no gap during market hours
   where the stream was disconnected for more than one backoff cap without
   reconnecting.
4. `latency_seconds_median` is recorded, so the latency gain is measured and
   not assumed.

That change must keep polling as a reconciliation backstop, must go through
the OMS writer instead of a second fill writer, and must be validated against
`oms_invariants`, `oms_lifecycle_parity` and `replay_live_parity_gate`.
