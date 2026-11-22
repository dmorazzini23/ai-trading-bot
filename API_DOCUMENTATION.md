# API Documentation

This file documents the HTTP routes that are present in the current Flask app
created by `ai_trading.app:create_app()`.

## Base URLs

Main API/runtime surface:

```text
http://127.0.0.1:9001
```

Standalone health app:

```text
http://127.0.0.1:${HEALTHCHECK_PORT:-8081}
```

The standalone health app is launched with:

```bash
RUN_HEALTHCHECK=1 python3 -m ai_trading.app
```

## Health Routes

### `GET /health`

Lightweight liveness payload with Alpaca diagnostics.

Example:

```bash
curl -s http://127.0.0.1:9001/health | jq .
```

Representative response:

```json
{
  "ok": true,
  "service": "ai-trading",
  "status": "service",
  "timestamp": "2026-04-16T04:42:22.000Z",
  "alpaca": {
    "sdk_ok": true,
    "initialized": false,
    "client_attached": false,
    "has_key": true,
    "has_secret": true,
    "base_url": "https://paper-api.alpaca.markets",
    "paper": true,
    "shadow_mode": false
  }
}
```

### `GET /healthz`

Canonical health payload shared by the API and health-service surfaces.

Example:

```bash
curl -s http://127.0.0.1:9001/healthz | jq .
```

Representative response fields:

```json
{
  "ok": true,
  "service": "ai-trading",
  "status": "service",
  "timestamp": "2026-04-16T04:42:22.000Z",
  "fallback_active": false,
  "data_provider": {
    "primary": "alpaca",
    "active": "alpaca"
  },
  "broker_connectivity": {
    "connected": false,
    "status": "unknown"
  },
  "quotes_status": {
    "allowed": true
  },
  "drawdown_breaker": {
    "available": true,
    "state": "closed",
    "peak_equity": 101250.5,
    "peak_set_at": "2026-11-20T15:31:04.512345+00:00",
    "current_drawdown": 0.012,
    "trip_threshold": 0.08,
    "reset_threshold": 0.064,
    "tripped_at": null,
    "last_reading_at": "2026-11-21T20:59:01.123456+00:00",
    "state_load": "restored",
    "persisted_state_loaded": true,
    "load_failure_reason": null,
    "last_write_failed": false
  }
}
```

Notes:

- `drawdown_breaker` is informational only and never changes `ok`, `status` or
  the HTTP code. The account id is never included.
- `state` is `closed`, `open`, or `unknown`. The saved state is read on the
  first equity reading of a market cycle, so after a restart `state` is
  `unknown` (not `closed`) until then, even if the saved file says open.
- `state_load` is `not_configured` (no state file), `pending` (not read yet, or
  the file was unreadable and is being retried; `load_failure_reason` is
  `unreadable` in the latter case), `restored`, `fresh` (no file; a normal first
  start), or `failed` (file rejected, fresh start; `load_failure_reason` is e.g.
  `corrupt`, `account_mismatch`, `schema_mismatch`, `peak_invalid`).
  `persisted_state_loaded` is true only for `restored`.
- `last_write_failed` is true after a failed state-file write and false again
  after the next successful write. `last_reading_at` is when the last accepted
  equity reading arrived, so a stale `current_drawdown` is visible; it and
  `peak_equity`/`current_drawdown` are `null` until the first reading.
- The section is `{"available": false, "reason": "not_initialized" |
  "snapshot_failed"}` when the breaker is absent or cannot be read.
  `not_initialized` is what the main service shows until the bot context is
  first built. The standalone health app (`RUN_HEALTHCHECK=1 python -m
  ai_trading.app`) is a separate process and always shows `not_initialized`.
  The snapshot reports in-memory state and never reads the state file.
- The handler is designed to degrade gracefully rather than raising uncaught
  exceptions.
- Health payloads may report `"status": "degraded"` while still returning HTTP
  `200`.

### `GET /metrics`

Prometheus-compatible metrics from the active runtime registry.

Example:

```bash
curl -s http://127.0.0.1:9001/metrics
```

Behavior:

- Returns `200` with Prometheus text when metrics are available.
- Returns `501` with `metrics unavailable` when the metrics backend is not
  available.

## Diagnostics Route

### `GET /diag`

Environment and Alpaca diagnostics from `ai_trading.diagnostics.env_diag`.

Example:

```bash
curl -s http://127.0.0.1:9001/diag | jq .
```

Representative response:

```json
{
  "alpaca": {
    "configured_feed": "iex",
    "data_base_url": "https://data.alpaca.markets",
    "environment": "paper",
    "has_key": true,
    "has_secret": true,
    "paper": true,
    "shadow_mode": false,
    "trading_base_url": "https://paper-api.alpaca.markets"
  }
}
```

## Operator Routes

### `GET /operator/presets`

Returns the available guarded operator presets.

Example:

```bash
curl -s http://127.0.0.1:9001/operator/presets | jq .
```

Representative response:

```json
{
  "ok": true,
  "presets": [
    {
      "name": "conservative",
      "trading_mode": "conservative",
      "capital_cap": 0.03
    },
    {
      "name": "balanced",
      "trading_mode": "balanced",
      "capital_cap": 0.06
    }
  ]
}
```

### `GET /operator/plan`

Returns the default guarded plan for the `balanced` preset.

```bash
curl -s http://127.0.0.1:9001/operator/plan | jq .
```

Representative response:

```json
{
  "ok": true,
  "plan": {
    "name": "balanced",
    "trading_mode": "balanced",
    "capital_cap": 0.06,
    "max_positions": 10
  }
}
```

### `POST /operator/plan`

Validates a requested preset and optional overrides.

Example request:

```bash
curl -s \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"preset":"balanced","overrides":{"max_positions":8,"capital_cap":0.05}}' \
  http://127.0.0.1:9001/operator/plan | jq .
```

Success response:

```json
{
  "ok": true,
  "plan": {
    "name": "balanced",
    "max_positions": 8,
    "capital_cap": 0.05
  }
}
```

Validation failure response:

```json
{
  "ok": false,
  "error": "capital_cap must be between ..."
}
```

### `GET /operator/control-plane`

Returns a consolidated runtime control-plane snapshot.

```bash
curl -s http://127.0.0.1:9001/operator/control-plane | jq .
```

Representative top-level snapshot fields:

```json
{
  "ok": true,
  "snapshot": {
    "service": "ai-trading",
    "rollout": {},
    "broker_health": {},
    "execution_quality": {},
    "manual_overrides": {},
    "governance": {}
  }
}
```

## Not Currently Documented As Public API

The current repository does not ship a canonical public websocket server,
browser SDK, or `ai_trading.client` module. Older docs that referenced
`/ws/data`, `/ws/trades`, or `TradingBotClient` were removed because those
surfaces do not exist in the current app factory.
