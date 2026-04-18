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
- `POST /operator/plan`
- `GET /operator/control-plane`

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
- `ai_trading.strategies.regime_detection` is the broader pandas/DataFrame
  market-regime analysis module used for rich analysis outputs.
- `ai_trading.strategies.regime_detector` is the dynamic-threshold and
  canonical `MarketRegime` module used by live-ish portfolio and signal flows.
- `ai_trading.tools.seed_trade_history` uses the packaged JSON seed file
  `ai_trading/defaults/trade_history.seed.json`, while runtime trade-history
  persistence is controlled separately through `AI_TRADING_TRADE_HISTORY_PATH`.

## Operational Constraints

- `zoneinfo` only; `pytz` is not part of the runtime path.
- Structured logging only.
- No compatibility shims in current runtime policy.
- The main runtime refuses to start when another healthy API is already bound
  to the configured API port.
