# Deployment Guide: Alpaca Feed Selection

## Intraday Feed Options

The trading bot supports multiple Alpaca intraday data feeds. Configure the
feed via the `DATA_FEED_INTRADAY` environment variable:

| Value | Description | SIP entitlement required |
|-------|-------------|--------------------------|
| `iex` | Default retail feed (15-minute IEX snapshot). | No |
| `sip` | Consolidated SIP tape (full NBBO coverage). | **Yes** |

### Required Environment Variables

Regardless of feed selection the following variables must be present at
startup:

* `ALPACA_API_KEY`
* `ALPACA_SECRET_KEY`
* `DATA_FEED_INTRADAY`

When `DATA_FEED_INTRADAY=sip`, Alpaca must confirm SIP access via one of:

* `ALPACA_ALLOW_SIP=1`
* `ALPACA_HAS_SIP=1`

If neither flag is set the service now fails fast with an actionable error so
operators can request entitlements before launch.

### Optional Overrides

* `ALPACA_FEED_FAILOVER` — Comma-separated backup order for Alpaca data feeds.
* `AI_TRADING_HALT_FLAG_PATH` — Override the default `halt.flag` location used
  by provider safe-mode.

### Troubleshooting SIP flags

`ALPACA_SIP_UNAUTHORIZED=1` only disables the SIP path when
`DATA_FEED_INTRADAY=sip`. Deployments running with `DATA_FEED_INTRADAY=iex`
ignore this flag so primary IEX orders continue flowing. Clear the flag once
SIP entitlements are restored before switching the intraday feed to `sip`.

### Validation Checklist

1. Export the required variables in the deployment environment.
2. Run `pip show alpaca-trade-api` to confirm runtime pins `3.2.0`.
3. Start the service and watch for
   `TRADING_PARAMS_VALIDATED`/`DATA_PROVIDER_READY` logs.
4. For SIP, call `alpaca-proxy/data/v2/stocks/AAPL/bars` with the deployment
   credentials to verify the consolidated feed before enabling live trading.

## API Serving Options

The default `ai-trading.service` unit launches the trader and its built-in Flask API in a single process. This keeps operational overhead low and is the recommended mode for lab environments or managed rollouts where the systemd unit supervises everything.

For production environments that prefer a hardened HTTP stack, deploy the companion `packaging/systemd/ai-trading-api.service`. It runs `gunicorn --workers 2 --threads 4 --bind 0.0.0.0:9001 'ai_trading.app:create_app()'`, letting systemd supervise a dedicated API process while the trading loop continues to run under `ai-trading.service`. Use this split when API uptime requirements differ from the trading loop, or when an ingress controller expects a WSGI server instead of Flask's development server.

When both services are active, ensure probes and load balancers target the Gunicorn port (`9001` by default) and keep the single-process unit available for scenarios where a combined deployment is simpler (edge devices, QA sandboxes, etc.).
