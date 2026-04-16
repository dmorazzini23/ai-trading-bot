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
  }
}
```

Notes:

- The handler is designed to degrade gracefully rather than raising uncaught
  exceptions.
- Health payloads may report `"status": "degraded"` while still returning HTTP
  `200`.

### `GET /metrics`

Prometheus metrics from the active runtime registry.

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

Environment and Alpaca diagnostics from `ai_trading.diagnostics.http_diag`.

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
