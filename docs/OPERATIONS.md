## Operations

Target OS: **Ubuntu 24.04**.

### Service Management

```bash
sudo systemctl start ai-trading.service
sudo systemctl stop ai-trading.service
sudo systemctl restart ai-trading.service
sudo systemctl status ai-trading.service
journalctl -u ai-trading.service -n 200 --no-pager
curl -s http://127.0.0.1:9001/healthz
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:9001/metrics
```

The packaged `ai-trading.service` runs the main API on `:9001` and, by
default, exposes `/healthz` and `/metrics` on that same API port.

Use `RUN_HEALTHCHECK=1 python -m ai_trading.app` only when you want the
standalone lightweight health app. In that mode the standalone health server
binds to `HEALTHCHECK_PORT` (default `8081`). In the main runtime path,
`RUN_HEALTHCHECK=1` requires `HEALTHCHECK_PORT != API_PORT`.

### Singleton Guard

The service refuses to start when another healthy `ai-trading` API is already
bound to `API_PORT` (`9001` by default). On startup it probes
`http://127.0.0.1:<API_PORT>/healthz`; if that probe returns
`service=ai-trading`, the new process logs `API_PORT_HEALTHY_ELSEWHERE` and
aborts rather than starting a duplicate trading loop.

### API Port Startup Behavior

`API_PORT_WAIT_SECONDS` defaults to **30 seconds**. Startup waits up to that
window for the API socket to become free. If the port stays busy, the process
exits with status `98`, and the packaged systemd unit uses
`RestartPreventExitStatus=98` so operators can resolve the conflict explicitly.

### Important Paths

- `TRADE_LOG_PATH` defaults to `/var/log/ai-trading-bot/trades.jsonl` unless
  overridden.
- The packaged systemd unit pins `TRADE_LOG_PATH` to
  `/home/aiuser/ai-trading-bot/logs/trades.jsonl`.
- Writable runtime directories are controlled by:
  `AI_TRADING_DATA_DIR`, `AI_TRADING_CACHE_DIR`, `AI_TRADING_LOG_DIR`,
  `AI_TRADING_MODELS_DIR`, and `AI_TRADING_OUTPUT_DIR`.

### Market-Data Incident Checks

1. Confirm the runtime SDK pin:
   `python3 -c "import importlib.metadata as m; print(m.version('alpaca-py'))"`
2. Confirm the current feed config:
   `python3 -c "from ai_trading.config.management import get_env; print(get_env('ALPACA_DATA_FEED'))"`
3. Check the control-plane snapshot:
   `curl -s http://127.0.0.1:9001/operator/control-plane | jq .`
4. Check environment diagnostics:
   `curl -s http://127.0.0.1:9001/diag | jq .`

### Provider Safe-Mode

Repeated Alpaca failures can trigger provider safe-mode:

1. A provider outage alert is emitted.
2. The halt flag is written to `HALT_FLAG_PATH`/`AI_TRADING_HALT_FLAG_PATH`.
3. The provider monitor backs the provider off.
4. New orders are blocked until the halt condition clears.

Resume only after Alpaca data is healthy again and the halt condition has been
cleared intentionally.
