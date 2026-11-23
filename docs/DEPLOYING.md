# Deployment Guide

## Runtime Targets

- Python `3.12.3`
- API on `0.0.0.0:9001`
- Standalone health app on `HEALTHCHECK_PORT` (default `8081`)
- Runtime SDK pin: `alpaca-py==0.42.1`
- Timezone handling: stdlib `zoneinfo` only

## Required Environment Variables

Current startup validation requires:

- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `ALPACA_TRADING_BASE_URL`
- `ALPACA_DATA_FEED`
- `WEBHOOK_SECRET`
- `AI_TRADING_CAPITAL_CAP`
- `DOLLAR_RISK_LIMIT`

Deprecated and rejected (startup fails when any is set). The full list, with
the canonical key for each, is the table in [../DEPLOYING.md](../DEPLOYING.md)
and `_CANONICAL_ENV_MAP` in `ai_trading/config/management.py`. It includes:

- `ALPACA_API_URL`, `ALPACA_BASE_URL` (use `ALPACA_TRADING_BASE_URL`)
- `ALPACA_DATA_URL` (use `ALPACA_DATA_BASE_URL`)
- `TRADING_MODE`, `CAPITAL_CAP`, `DAILY_LOSS_LIMIT`, `MAX_POSITION_SIZE`,
  `TAKE_PROFIT_FACTOR`, `BUY_THRESHOLD`, `CONF_THRESHOLD`, `MIN_CONFIDENCE`,
  `KELLY_FRACTION`, `KELLY_FRACTION_MAX`, `SIGNAL_CONFIRMATION_BARS`,
  `NEWS_API_KEY`

`DATA_FEED` is deprecated but not rejected: it is still read as an alias for
`ALPACA_DATA_FEED`. Use `ALPACA_DATA_FEED`.

## Feed Selection

Two related settings exist:

- `ALPACA_DATA_FEED`: required Alpaca market-data preference. Valid values:
  `iex`, `sip`.
- `DATA_FEED_INTRADAY`: execution-pricing preference. Valid values:
  `iex`, `sip`, `finnhub`.

If `DATA_FEED_INTRADAY=sip` or `ALPACA_EXECUTION_FEED=sip`, the runtime expects
SIP entitlements and will fail fast without them.

Optional feed controls:

- `ALPACA_ALLOW_SIP=1`
- `ALPACA_HAS_SIP=1`
- `ALPACA_FEED_FAILOVER=sip,iex`
- `ALPACA_EMPTY_TO_BACKUP=1`
- `BACKUP_DATA_PROVIDER=yahoo|finnhub|finnhub_low_latency|none` (default
  `yahoo`, or `none` when `EXECUTION_MODE=live`; live mode does not permit
  Yahoo)

Production execution note:

- Live opening orders fail closed unless they have an approved real-time broker
  NBBO execution quote. Yahoo is not permitted as a live execution backup data
  source, and live execution now blocks Yahoo fallback in quote and minute-data
  recovery paths.
- Live execution defaults are now strict even if permissive quote env vars are
  omitted: realtime NBBO is required, degraded quotes do not downgrade to
  market, fallback pricing is disabled, and limit orders require NBBO unless an
  explicit non-production fallback-quote override is enabled.
- Legacy non-netting live execution is blocked outside test runtimes.

## Packaged Services

Before deploying a candidate, run the read-only release identity preflight
described in [RELEASE_IDENTITY.md](RELEASE_IDENTITY.md). It compares the tested
commit, effective config/profile, applied migration and configured model bytes
against a reviewed release specification. The installed packaged unit enforces
checks before and after Alembic migration and refuses startup without a valid
release specification at `/var/lib/ai-trading-bot/runtime/release_spec.json`.
The spec's `tested_commit_sha` must equal the checkout's `HEAD`, the checkout
must be clean, and its `resolved_config_hash` must equal the hash the new code
computes. Any ConfigSpec change alters `resolved_config_hash` and blocks
startup until the spec is updated. The checks and the safe deploy order are in
"Release Identity Check" in [OPERATIONS.md](OPERATIONS.md).

Historical record, not the current state: on October 8, 2026 19:33 Pacific,
after the host maintenance reboot, a paper start with CI-tested commit
`1853464dc` was verified with both installed release checks passing. The
broker was closed,
ACTIVE and flat before restart; post-start broker freshness and OMS
consistency passed at 19:40 Pacific, with zero positions/active orders and
automatic restarts. The service has been redeployed since; the currently
deployed commit is recorded in [CODEX_HANDOFF.md](CODEX_HANDOFF.md).
For each subsequent candidate, review a
passing report with CI, broker exposure and rollback evidence before updating
the release specification or restarting. Do not restart during market hours.

Canonical production service:

```bash
sudo cp packaging/systemd/ai-trading.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-trading.service
```

Institutional production deployment uses `ai-trading.service` as the single
authoritative topology for trading, API, health, and metrics on `:9001`.

The packaged services sync `/home/aiuser/ai-trading-bot/.env` into
`/run/ai-trading-bot/ai-trading-runtime.env` during service startup. Edit the
repo `.env`, then restart the service; do not maintain a separate packaged env
copy under `/etc/ai-trading-bot`.

## Health Surfaces

Main runtime:

- `http://127.0.0.1:9001/health`
- `http://127.0.0.1:9001/healthz`
- `http://127.0.0.1:9001/metrics`

Standalone health app:

```bash
RUN_HEALTHCHECK=1 python3 -m ai_trading.app &
curl -s http://127.0.0.1:${HEALTHCHECK_PORT:-8081}/healthz
```

When launching the standalone health app, `HEALTHCHECK_PORT` must differ from
`API_PORT`.

`/metrics` answers loopback callers by default; remote callers need
`AI_TRADING_METRICS_BEARER_TOKEN` or an address in
`AI_TRADING_METRICS_ALLOWED_REMOTE_ADDRS` (see
[../API_DOCUMENTATION.md](../API_DOCUMENTATION.md)).

## API Port Startup

The API binds to `API_PORT` (default `9001`). Startup waits up to
`API_PORT_WAIT_SECONDS` (default 30 seconds) for the port to become free. If
it stays busy, the process exits with status `98`, and the packaged unit's
`RestartPreventExitStatus=98` stops systemd from restarting it.

## Verification Checklist

1. `python3 -c "import importlib.metadata as m; assert m.version('alpaca-py') == '0.42.1'"`
2. `python3 -m ai_trading.tools.env_validate`
3. `journalctl -u ai-trading.service -n 200 --no-pager`
4. `curl -s http://127.0.0.1:9001/healthz | jq .`
5. `curl -s -H 'Authorization: Bearer <operator-token>' -H 'X-AI-Trading-Operator-Id: <operator-id>' http://127.0.0.1:9001/operator/control-plane | jq .`

## Operator Access

- Configure operator credentials with `AI_TRADING_OPERATOR_TOKEN_MAP`, not a
  shared bearer token.
- `/diag` and all `/operator/*` endpoints require an authenticated operator
  token plus `X-AI-Trading-Operator-Id`.
- The drawdown circuit breaker (saved state, staying tripped across restarts,
  and the operator clear procedure) and the position backstop are described in
  [OPERATIONS.md](OPERATIONS.md).

## Additional Runtime Hardening

- Pretrade pacing is persisted by default outside tests at
  `runtime/pretrade_rate_limiter.db`. Override with
  `AI_TRADING_PRETRADE_RATE_LIMITER_PATH` only when pointing to a shared durable
  runtime volume.
- Generic pickle/cloudpickle/dill model deserialization is retired from the
  supported runtime/model-registry path.
- The rich model registry only accepts JSON-safe
  inline artifacts or explicit approved artifact paths.
- The legacy database manager has been removed; runtime durability uses
  OMS-backed persistence only.
