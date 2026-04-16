# Deployment Guide

This repository's current production path is the packaged `systemd` workflow.
Older `:5000`-based Docker and proxy examples were removed from this document
because the active runtime is now standardized on:

- API on `:9001`
- Standalone health app on `HEALTHCHECK_PORT` (default `:8081`)
- `alpaca-py==0.42.1`
- canonical broker endpoint env: `ALPACA_TRADING_BASE_URL`

## Recommended Deployment Path

1. Install the repo into a Python 3.12 virtualenv.
2. Configure the required environment variables in `.env`.
3. Install `packaging/systemd/ai-trading.service`.
4. Optionally install `packaging/systemd/ai-trading-api.service` if you want a
   dedicated Gunicorn API process.

For the concrete deployment steps, use:

- [DEPLOYING.md](DEPLOYING.md)
- [docs/DEPLOYING.md](docs/DEPLOYING.md)
- [docs/OPERATIONS.md](docs/OPERATIONS.md)

## Required Environment Variables

Current startup validation requires:

- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `ALPACA_TRADING_BASE_URL`
- `ALPACA_DATA_FEED`
- `WEBHOOK_SECRET`
- `AI_TRADING_CAPITAL_CAP`
- `DOLLAR_RISK_LIMIT`

Deprecated and rejected:

- `ALPACA_API_URL`
- `ALPACA_BASE_URL`

## Health and Verification

Main runtime:

```bash
curl -s http://127.0.0.1:9001/healthz | jq .
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:9001/metrics
curl -s http://127.0.0.1:9001/operator/control-plane | jq .
```

Standalone health app:

```bash
RUN_HEALTHCHECK=1 python3 -m ai_trading.app &
curl -s http://127.0.0.1:${HEALTHCHECK_PORT:-8081}/healthz | jq .
```

## Notes

- `RUN_HEALTHCHECK=1` is for the standalone health app. In the main runtime
  path, that mode requires `HEALTHCHECK_PORT != API_PORT`.
- `API_PORT_WAIT_SECONDS` defaults to `30`.
- The main runtime refuses to start if a healthy `ai-trading` API is already
  bound to the configured API port.
