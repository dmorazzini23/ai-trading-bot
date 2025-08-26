## Deploying the AI Trading Bot (systemd)

Target OS: **Ubuntu 24.04**.

Use the canonical unit at `packaging/systemd/ai-trading.service`.

```bash
sudo cp packaging/systemd/ai-trading.service /etc/systemd/system/ai-trading.service
sudo systemctl daemon-reload
sudo systemctl enable --now ai-trading.service
sudo systemctl restart ai-trading.service
sudo systemctl status ai-trading.service
```

Check logs and health:

```bash
journalctl -u ai-trading.service -n 200 --no-pager
curl -sf http://127.0.0.1:$HEALTHCHECK_PORT/healthz
```

Configure environment in `/home/aiuser/ai-trading-bot/.env` (loaded with `override=True` at startup) and ensure PATH in the unit points to your venv.

### Required environment variables

`ai_trading.config.management.validate_required_env()` ensures these keys are
present before the service starts:

| Key | Purpose |
| --- | --- |
| `ALPACA_API_KEY` | Alpaca API authentication |
| `ALPACA_SECRET_KEY` | Alpaca API authentication |
| `ALPACA_BASE_URL` | Broker endpoint URL |
| `WEBHOOK_SECRET` | Protects inbound webhooks |

If any are missing or empty the process exits with a `RuntimeError` listing the
missing keys; values are masked in logs and exceptions.

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
