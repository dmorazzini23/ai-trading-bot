## Deploying the AI Trading Bot (systemd)

Target OS: **Ubuntu 24.04**.

Use the canonical unit at `packaging/systemd/ai-trading.service`.

```bash
sudo cp packaging/systemd/ai-trading.service /etc/systemd/system/ai-trading.service
sudo cp packaging/systemd/ai-trading.timer /etc/systemd/system/ai-trading.timer
sudo systemctl daemon-reload
# Confirm host timezone (service timer uses America/New_York)
timedatectl | grep 'Time zone'
# Enable timer to run the bot 09:30-16:00 US/Eastern
sudo systemctl enable --now ai-trading.timer
sudo systemctl status ai-trading.service
```

The timer schedules the bot to start at market open and the service exits automatically after 6.5 hours (16:00 US/Eastern). On normal completion the process returns exit code `0` so systemd records a clean shutdown.

If the bot is started manually outside regular hours, it waits until the next
NYSE session before trading. Set `ALLOW_AFTER_HOURS=1` to disable this wait
when running tests or after-hours experiments.

Check logs and health:

```bash
journalctl -u ai-trading.service -n 200 --no-pager
curl -sf http://127.0.0.1:$HEALTHCHECK_PORT/healthz
```

For local verification without systemd:

```bash
RUN_HEALTHCHECK=1 python -m ai_trading.app &
curl -sf http://127.0.0.1:$HEALTHCHECK_PORT/healthz
```

Configure environment in `/home/aiuser/ai-trading-bot/.env` (loaded with `override=True` at startup) and ensure PATH in the unit points to your venv.

Startup runs an import preflight and will exit if the `alpaca-py` package is missing.

### Required environment variables

`ai_trading.config.management.validate_required_env()` ensures these keys are
present before the service starts:

| Key | Purpose |
| --- | --- |
| `ALPACA_API_KEY` | Alpaca API authentication |
| `ALPACA_SECRET_KEY` | Alpaca API authentication |
| `ALPACA_API_URL` / `ALPACA_BASE_URL` | Broker endpoint URL (either variable is accepted and logged) |
| `ALPACA_DATA_FEED` | Market data feed selection |
| `WEBHOOK_SECRET` | Protects inbound webhooks |
| `CAPITAL_CAP` | Maximum portfolio allocation |
| `DOLLAR_RISK_LIMIT` | Maximum per-trade dollar risk |

Either `ALPACA_API_URL` or `ALPACA_BASE_URL` may be set; the service resolves the
alias and logs the effective URL during startup.

If any are missing or empty the process exits with a `RuntimeError` listing the
missing keys; values are masked in logs and exceptions. During health server
startup the validation result is cached and `/healthz` reuses it, returning
`{"ok": false, "error": "..."}` while still responding with HTTP 200.

### Optional risk parameters

| Key | Purpose | Default |
| --- | --- | --- |
| `AI_TRADING_CONF_THRESHOLD` | Minimum model confidence required before acting | 0.75 |

### Persistent directories

The service writes state, cache, logs, models, and run outputs to paths governed by the environment variables `AI_TRADING_DATA_DIR`, `AI_TRADING_CACHE_DIR`, `AI_TRADING_LOG_DIR`, `AI_TRADING_MODELS_DIR`, and `AI_TRADING_OUTPUT_DIR`.
Each directory must exist and be writable by the service user with **0700** permissions.

```bash
sudo install -d -m 700 -o aiuser -g aiuser \
  /var/lib/ai-trading-bot \
  /var/lib/ai-trading-bot/models \
  /var/lib/ai-trading-bot/output \
  /var/cache/ai-trading-bot \
  /var/log/ai-trading-bot
```

Mount these locations or set the variables above so data persists across restarts.

### Health endpoints & env

Set `RUN_HEALTHCHECK=1` to expose `/healthz` and `/metrics` on the port defined by the `HEALTHCHECK_PORT` environment variable (default **9001**).

| Key | Purpose |
| --- | --- |
| `RUN_HEALTHCHECK` | Enable lightweight Flask health server |
| `HEALTHCHECK_PORT` | Port for `/healthz` and `/metrics` |

### CLI

`ai-trade`, `ai-backtest`, and `ai-health` share flags:

| Flag | Description |
| ---- | ----------- |
| `--dry-run` | Exit after importing modules |
| `--once` | Run a single iteration then exit |
| `--interval SECONDS` | Sleep between iterations |
| `--paper` / `--live` | Select paper (default) or live trading |

### Operational checklist

1. `sudo systemctl restart ai-trading.service`
2. `journalctl -u ai-trading.service -n 200 --no-pager`
3. `curl -sf http://127.0.0.1:$HEALTHCHECK_PORT/healthz`
4. Roll back to previous release and restart if the health check fails
