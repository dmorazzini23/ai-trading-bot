## Deploying the AI Trading Bot (systemd)

Target OS: **Ubuntu 24.04**.

Use the canonical unit at `packaging/systemd/ai-trading.service`.

```bash
sudo cp packaging/systemd/ai-trading.service /etc/systemd/system/ai-trading.service
sudo cp packaging/systemd/ai-trading.timer /etc/systemd/system/ai-trading.timer
sudo systemctl daemon-reload
sudo systemctl enable --now ai-trading.timer
sudo systemctl status ai-trading.service
```

The timer schedules the bot around the market session. If the service starts
outside market hours, the runtime waits for the next NYSE session unless
`ALLOW_AFTER_HOURS=1` is set.

## Environment Sync

The packaged units treat `/home/aiuser/ai-trading-bot/.env` as the editable
source of truth. Each service start runs `scripts/sync_env_runtime.sh`, which
renders `/run/ai-trading-bot/ai-trading-runtime.env`; the process then loads
that runtime file. After the systemd units are deployed, normal config changes
only require editing `.env` and restarting the relevant service.

There is no repo-local runtime env file in the production or operator workflow.
Manual commands should either read `.env` directly or use the generated
`/run/ai-trading-bot/ai-trading-runtime.env` when the service is installed.

## Health Checks

Packaged main service:

```bash
journalctl -u ai-trading.service -n 200 --no-pager
curl -s http://127.0.0.1:9001/healthz | jq .
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:9001/metrics
```

Standalone health app:

```bash
RUN_HEALTHCHECK=1 python3 -m ai_trading.app &
curl -s http://127.0.0.1:${HEALTHCHECK_PORT:-8081}/healthz | jq .
```

`RUN_HEALTHCHECK=1` is only for the standalone health app. In the main runtime
path, setting it requires `HEALTHCHECK_PORT` to differ from `API_PORT`.

## Required Environment Variables

`ai_trading.config.management.validate_required_env()` currently requires:

| Key | Purpose |
| --- | --- |
| `ALPACA_API_KEY` | Alpaca API authentication |
| `ALPACA_SECRET_KEY` | Alpaca API authentication |
| `ALPACA_TRADING_BASE_URL` | Canonical Alpaca trading endpoint |
| `ALPACA_DATA_FEED` | Market-data feed selection |
| `WEBHOOK_SECRET` | Protects inbound webhooks |
| `AI_TRADING_CAPITAL_CAP` | Capital allocation guardrail |
| `DOLLAR_RISK_LIMIT` | Per-position risk guardrail |

Rejected deprecated aliases:

- `ALPACA_API_URL`
- `ALPACA_BASE_URL`

## Feed Configuration

- `ALPACA_DATA_FEED`: required Alpaca market-data preference (`iex` or `sip`)
- `DATA_FEED_INTRADAY`: intraday execution-pricing preference (`iex`, `sip`, `finnhub`)
- `ALPACA_EXECUTION_FEED`: explicit execution-pricing override

SIP-related controls:

- `ALPACA_ALLOW_SIP=1`
- `ALPACA_HAS_SIP=1`
- `ALPACA_FEED_FAILOVER=sip,iex`
- `ALPACA_EMPTY_TO_BACKUP=1`

Backup market-data providers are not approved live opening-order quote sources.
Live opening orders fail closed unless they have a real-time broker NBBO quote.
Live execution defaults are strict: realtime NBBO is required, degraded quotes
do not downgrade to market, fallback pricing is disabled, and limit orders
require NBBO unless a non-production fallback-quote override is explicitly
enabled.
Legacy non-netting live execution is blocked outside test runtimes.

## API Port Fail-Fast Semantics

The primary Flask API binds to `API_PORT` (default **9001**). Startup waits up
to `API_PORT_WAIT_SECONDS` (default **30s**) for the port to become free. If it
stays busy, the process exits with status **98** and the packaged systemd unit
uses `RestartPreventExitStatus=98`.

## Operational Checklist

1. `python3 -c "import importlib.metadata as m; assert m.version('alpaca-py') == '0.42.1'"`
2. `python3 -m ai_trading.tools.env_validate`
3. `sudo systemctl restart ai-trading.service`
4. `journalctl -u ai-trading.service -n 200 --no-pager`
5. `curl -s http://127.0.0.1:9001/healthz | jq .`

## Research Automation Timers

After installing the main service, install the after-hours research timers:

```bash
sudo cp packaging/systemd/ai-trading-research-*.service /etc/systemd/system/
sudo cp packaging/systemd/ai-trading-research-*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-trading-research-daily.timer
sudo systemctl enable --now ai-trading-research-weekly.timer
sudo systemctl enable --now ai-trading-research-monthly.timer
systemctl list-timers 'ai-trading-research-*' --no-pager
```

The timers write research artifacts under
`/var/lib/ai-trading-bot/runtime/research_reports/`. They are evidence builders
only; model promotion and live-money cutover remain manual gated workflows.

## Additional Hardening Defaults

- Pretrade pacing persists to `runtime/pretrade_rate_limiter.db` by default outside tests.
- Generic pickle/cloudpickle/dill model deserialization is retired from the
  supported runtime/model-registry path.
- Live mode does not permit Yahoo as a backup market-data provider.
- Legacy database access is retired outside tests; runtime durability should use
  OMS-backed stores only.
- Operator access should use `AI_TRADING_OPERATOR_TOKEN_MAP` with per-operator
  credentials rather than a shared bearer token.
