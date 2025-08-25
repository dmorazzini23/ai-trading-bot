## Operations

Target OS: **Ubuntu 24.04**.

### Service Management

```bash
sudo systemctl start ai-trading.service
sudo systemctl stop ai-trading.service
sudo systemctl restart ai-trading.service
sudo systemctl status ai-trading.service
journalctl -u ai-trading.service -n 200 --no-pager
curl -s http://127.0.0.1:9001/health  # JSON, never 500
```

### Paths & default files
- Trade log file defaults to `<repo>/logs/trades.jsonl` when `TRADE_LOG_PATH` is not set. The directory is auto-created.
- Empty model path disables ML quietly. Set `MODEL_PATH` to enable.
