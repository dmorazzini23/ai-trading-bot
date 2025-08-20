## Deploying the AI Trading Bot (systemd)

Use the canonical unit at `packaging/systemd/ai-trading.service`.

```bash
sudo cp packaging/systemd/ai-trading.service /etc/systemd/system/ai-trading.service
sudo systemctl daemon-reload
sudo systemctl enable --now ai-trading.service
```

Configure environment in `/home/aiuser/ai-trading-bot/.env` and ensure PATH in the unit points to your venv.

