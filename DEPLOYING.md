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
curl -s http://127.0.0.1:9001/health
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

