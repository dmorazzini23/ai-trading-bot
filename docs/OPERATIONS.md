## Operations

Target OS: **Ubuntu 24.04**.

### Service Management

```bash
sudo systemctl start ai-trading.service
sudo systemctl stop ai-trading.service
sudo systemctl restart ai-trading.service
sudo systemctl status ai-trading.service
journalctl -u ai-trading.service -n 200 --no-pager
curl -s http://127.0.0.1:9001/healthz  # JSON, never 500
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:9001/metrics  # 200 if enabled, else 501
```

Set `RUN_HEALTHCHECK=1` in the environment to enable the Flask endpoints.

### Paths & default files
- Trade log file defaults to `<repo>/logs/trades.jsonl` when `TRADE_LOG_PATH` is not set. The directory is auto-created.
- Empty model path disables ML quietly. Set `MODEL_PATH` to enable.

### Position sizing environment variables
Set the following to control position sizing:

- `CAPITAL_CAP`: fraction of equity usable per cycle.
- `DOLLAR_RISK_LIMIT`: fraction of equity at risk per position.
- `MAX_POSITION_SIZE`: absolute USD cap per position. Must be >0 (typically 1-10000). Values ≤0 raise a configuration error. If unset, the bot derives a value from `CAPITAL_CAP` and equity.
