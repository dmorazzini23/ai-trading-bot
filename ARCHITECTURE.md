# Architecture Overview

## Runtime

- Python `3.12.3`
- Main entrypoint: `python -m ai_trading`
- Main API: `ai_trading.app:create_app()`
- Main API port: `9001`
- Optional standalone health app: `python -m ai_trading.app` with
  `RUN_HEALTHCHECK=1` on `HEALTHCHECK_PORT` (default `8081`)
- Runtime SDK pin: `alpaca-py==0.42.1`

## Configuration

- Canonical runtime config access: `ai_trading.config.management`
- Required startup env includes:
  `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_TRADING_BASE_URL`,
  `ALPACA_DATA_FEED`, `WEBHOOK_SECRET`, `AI_TRADING_CAPITAL_CAP`,
  `DOLLAR_RISK_LIMIT`
- Deprecated and rejected env aliases include:
  `ALPACA_API_URL` and `ALPACA_BASE_URL`

## HTTP Surface

Current canonical routes from `create_app()`:

- `GET /health`
- `GET /healthz`
- `GET /metrics`
- `GET /diag`
- `GET /operator/presets`
- `GET /operator/plan`
- `GET /operator/control-plane`
- `GET /operator/control-plane/<section>` for the sections `rollout`,
  `broker-health`, `positions`, `open-orders`, `execution-quality`,
  `circuit-breakers`, `liveness`, `manual-overrides` and `services`
- `POST /operator/control-plane/manual-overrides`
- `DELETE /operator/control-plane/manual-overrides`
- `GET /operator/governance`
- `POST /operator/governance/approval`
- `POST /operator/governance/rollback`

`/health` and `/healthz` are registered through the shared
`ai_trading.health_payload.register_health_routes` helper. `/diag` and every
`/operator/*` route require operator authentication; `/metrics` has its own
access control. See `API_DOCUMENTATION.md`.

## Control Flow

1. `python -m ai_trading` validates config and startup invariants.
2. `ai_trading.main` starts the API server on `API_PORT`.
3. The scheduler loop coordinates data fetch, signal generation, risk checks,
   and execution through `ai_trading.core.bot_engine`.
4. Health payloads are built through `ai_trading.health_payload`, which keeps
   `/healthz` behavior canonical across entrypoints.

## Evaluation And Strategy Notes

- `ai_trading.strategies.backtester` is the fast research backtest path.
- `ai_trading.tools.offline_replay` is the production-faithful historical
  replay path.
- `ai_trading.strategies.regime_detector` is the dynamic-threshold regime
  detector used by `ai_trading.rebalancer` and `ai_trading.signals`. The `MarketRegime` enum itself
  is defined in `ai_trading.risk.adaptive_sizing`, which `regime_detector`
  imports.
- `ai_trading.tools.seed_trade_history` uses the packaged JSON seed file
  `ai_trading/defaults/trade_history.seed.json`, while runtime trade-history
  persistence is controlled separately through `AI_TRADING_TRADE_HISTORY_PATH`.

## Operational Constraints

- `zoneinfo` only; `pytz` is not part of the runtime path.
- Structured logging only.
- No compatibility shims in current runtime policy.
- The main runtime refuses to start when another healthy API is already bound
  to the configured API port.
- The packaged unit runs `ai_trading.tools.release_identity` before and after
  the Alembic migration and refuses to start when the release specification
  does not match the checkout, config, schema and model. See "Release identity
  check" in `docs/OPERATIONS.md`.
- The drawdown circuit breaker (`ai_trading.risk.circuit_breakers`) and the
  position backstop (`ai_trading.core.netting_position_backstop`) are
  described in `docs/OPERATIONS.md`.
