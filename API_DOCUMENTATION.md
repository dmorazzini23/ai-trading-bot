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

The standalone health app registers only `/healthz` and `/metrics`.

## Authentication

`/health` and `/healthz` are unauthenticated. `/metrics` has its own access
control (see below). `/diag` and every `/operator/*` route require operator
authentication and fail closed when it is not configured.

Each request must send:

- `Authorization: Bearer <token>`
- `X-AI-Trading-Operator-Id: <operator-id>` (`X-Operator-Id` is also accepted)

The examples below use `$OPERATOR_TOKEN` and `$OPERATOR_ID` as placeholders.

Settings keys (names only; see `docs/OPERATIONS.md`):

- `AI_TRADING_OPERATOR_TOKEN_MAP`: per-operator tokens. When it is set, the
  token must match the entry for the operator id.
- `AI_TRADING_OPERATOR_API_TOKEN` with `AI_TRADING_ALLOW_SHARED_OPERATOR_TOKEN`:
  a shared token. It is accepted only for read routes while
  `AI_TRADING_OPERATOR_READERS` is empty, and never for the mutating routes.
- Allowlists: `AI_TRADING_OPERATOR_READERS` (optional, read routes),
  `AI_TRADING_OPERATOR_OVERRIDE_OPERATORS` (manual-override `POST` and
  `DELETE`), `AI_TRADING_OPERATOR_APPROVERS` (governance approval) and
  `AI_TRADING_OPERATOR_ROLLBACK_OPERATORS` (governance rollback). The last
  three are required for their routes.

Error responses: `401` for a missing bearer token, `400` for a missing operator
id header, `403` for a rejected token or an operator outside the allowlist, and
`503` when authentication or a required allowlist is not configured.

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

Representative response fields (the payload has more top-level fields, for
example `service_state`, `attention_flags`, `operating_mode`,
`launch_profile`, `provider_authority` and `day_sleeve_model`):

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
- Returns `503` with `metrics temporarily unavailable` when generating the
  metrics text fails.

Access control:

- Callers whose remote address is in `AI_TRADING_METRICS_ALLOWED_REMOTE_ADDRS`
  (comma-separated addresses or CIDR networks; default `127.0.0.1,::1`) are
  allowed.
- Other callers must send `Authorization: Bearer <token>` matching
  `AI_TRADING_METRICS_BEARER_TOKEN`. Without a match the route returns `403`
  with `metrics access denied` and logs `PROM_METRICS_ACCESS_REJECTED`.
- `AI_TRADING_METRICS_ALLOW_UNAUTHENTICATED=1` turns the check off.

Remote example:

```bash
curl -s -H "Authorization: Bearer $METRICS_TOKEN" http://<host>:9001/metrics
```

## Diagnostics Route

### `GET /diag`

Environment and Alpaca diagnostics from `ai_trading.diagnostics.env_diag`.
Requires operator authentication.

Example:

```bash
curl -s \
  -H "Authorization: Bearer $OPERATOR_TOKEN" \
  -H "X-AI-Trading-Operator-Id: $OPERATOR_ID" \
  http://127.0.0.1:9001/diag | jq .
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

Every route in this section requires operator authentication (see
[Authentication](#authentication)).

### `GET /operator/presets`

Returns the available guarded operator presets.

Example:

```bash
curl -s \
  -H "Authorization: Bearer $OPERATOR_TOKEN" \
  -H "X-AI-Trading-Operator-Id: $OPERATOR_ID" \
  http://127.0.0.1:9001/operator/presets | jq .
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

Returns the default guarded plan for the `balanced` preset. This route is
`GET` only; there is no `POST /operator/plan`.

```bash
curl -s \
  -H "Authorization: Bearer $OPERATOR_TOKEN" \
  -H "X-AI-Trading-Operator-Id: $OPERATOR_ID" \
  http://127.0.0.1:9001/operator/plan | jq .
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

### `GET /operator/control-plane`

Returns a consolidated runtime control-plane snapshot.

```bash
curl -s \
  -H "Authorization: Bearer $OPERATOR_TOKEN" \
  -H "X-AI-Trading-Operator-Id: $OPERATOR_ID" \
  http://127.0.0.1:9001/operator/control-plane | jq .
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

### `GET /operator/control-plane/<section>`

Returns one control-plane section as `{"ok": true, "section": ..., "data":
...}`. Read auth. Sections: `rollout`, `broker-health`, `positions`,
`open-orders`, `execution-quality`, `circuit-breakers`, `liveness`,
`manual-overrides`, `services`.

```bash
curl -s \
  -H "Authorization: Bearer $OPERATOR_TOKEN" \
  -H "X-AI-Trading-Operator-Id: $OPERATOR_ID" \
  http://127.0.0.1:9001/operator/control-plane/broker-health | jq .
```

### `POST /operator/control-plane/manual-overrides`

Persists manual runtime overrides. Requires a per-operator token and an
operator listed in `AI_TRADING_OPERATOR_OVERRIDE_OPERATORS`. The JSON body
must contain `disabled_slices` (a list); `diagnostics` (an object) and
`source_updated_at` are optional. A missing or invalid body returns `400`.

```bash
curl -s -X POST \
  -H "Authorization: Bearer $OPERATOR_TOKEN" \
  -H "X-AI-Trading-Operator-Id: $OPERATOR_ID" \
  -H 'Content-Type: application/json' \
  -d '{"disabled_slices": []}' \
  http://127.0.0.1:9001/operator/control-plane/manual-overrides | jq .
```

### `DELETE /operator/control-plane/manual-overrides`

Clears the manual overrides. Same auth and allowlist as the `POST` route.

### `GET /operator/governance`

Returns the governance snapshot (approvals, scorecards and rollback audit).
Read auth.

```bash
curl -s \
  -H "Authorization: Bearer $OPERATOR_TOKEN" \
  -H "X-AI-Trading-Operator-Id: $OPERATOR_ID" \
  http://127.0.0.1:9001/operator/governance | jq .
```

### `POST /operator/governance/approval`

Records a governance approval or rejection checkpoint. Requires a per-operator
token and an operator listed in `AI_TRADING_OPERATOR_APPROVERS`. The JSON body
must contain `strategy` and `model_id` (otherwise `400`); `decision` (default
`approved`), `note` and `ticket` are optional. The approver is the
authenticated operator id.

### `POST /operator/governance/rollback`

Rolls a strategy back to its previous production model. Requires a
per-operator token and an operator listed in
`AI_TRADING_OPERATOR_ROLLBACK_OPERATORS`. The JSON body must contain
`strategy` (otherwise `400`); `reason` and `force` (default true) are
optional. Returns `200` when the rollback happened and `409` when it did not.

## Not Currently Documented As Public API

The current repository does not ship a canonical public websocket server,
browser SDK, or `ai_trading.client` module. Older docs that referenced
`/ws/data`, `/ws/trades`, or `TradingBotClient` were removed because those
surfaces do not exist in the current app factory.
