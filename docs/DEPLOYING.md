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

Deprecated and rejected:

- `ALPACA_API_URL`
- `ALPACA_BASE_URL`
- `DATA_FEED`

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
- `BACKUP_DATA_PROVIDER=yahoo|finnhub|finnhub_low_latency|none`

## Packaged Services

Primary trader + API:

```bash
sudo cp packaging/systemd/ai-trading.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-trading.service
```

Optional dedicated Gunicorn API:

```bash
sudo cp packaging/systemd/ai-trading-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-trading-api.service
```

The packaged Gunicorn service binds:

```bash
0.0.0.0:9001
```

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

## Verification Checklist

1. `python3 -c "import importlib.metadata as m; assert m.version('alpaca-py') == '0.42.1'"`
2. `python3 -m ai_trading.tools.env_validate`
3. `journalctl -u ai-trading.service -n 200 --no-pager`
4. `curl -s http://127.0.0.1:9001/healthz | jq .`
5. `curl -s http://127.0.0.1:9001/operator/control-plane | jq .`
