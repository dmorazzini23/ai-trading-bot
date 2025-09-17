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

### Singleton guard

The service refuses to start when another healthy `ai-trading` API is already
bound to the configured port (`9001` by default). On startup it probes
`http://127.0.0.1:<port>/healthz`; if it responds with `service=ai-trading`,
the new process logs `API_PORT_HEALTHY_ELSEWHERE` and exits with
`EADDRINUSE`. Operators should stop the existing `ai-trading.service` (or
release the port) before relaunching to avoid running duplicate trading loops.

### Paths & default files
- Trade log location is controlled by `TRADE_LOG_PATH` (or the legacy `AI_TRADING_TRADE_LOG_PATH`). The packaged systemd unit pins it to `/home/aiuser/ai-trading-bot/logs/trades.jsonl` and creates that directory with `0700` permissions for `aiuser`.
- Without an override the bot prefers `/var/log/ai-trading-bot/trades.jsonl`; if that directory—or an explicit override—cannot be created or written it automatically falls back to `./logs/trades.jsonl` relative to the working directory. The chosen directory is auto-created.
- The application initializes this trade log on startup via `ai_trading.core.bot_engine.get_trade_logger()` and trade execution lazily creates it if missing. Custom deployments should call it once if they bypass the standard entrypoint.
- Startup verifies this trade log path is writable, logs `TRADE_LOG_PATH_READY` with the resolved location, and exits if it cannot be created.
- Empty model path disables ML quietly. Set `MODEL_PATH` to enable.
- Override cache location with `AI_TRADING_CACHE_DIR` when the default `~/.cache/ai-trading-bot`
  path is not writable (for example, on read-only home directories). The application
  creates this directory with `0700` permissions during startup.

### Position sizing environment variables
Set the following to control position sizing:

- `CAPITAL_CAP`: fraction of equity usable per cycle.
- `DOLLAR_RISK_LIMIT`: fraction of equity at risk per position.
- `MAX_POSITION_SIZE`: absolute USD cap per position. Must be >0 (typically 1-10000). Ignored when `max_position_mode=AUTO`, where the bot derives a value from `CAPITAL_CAP` and equity.
- `AI_TRADING_MAX_POSITION_SIZE`: explicit override for deployments; must be positive and always takes precedence over dynamic sizing.
- `MAX_POSITION_EQUITY_FALLBACK`: equity assumed when deriving `MAX_POSITION_SIZE` but the real account equity cannot be fetched. Defaults to `200000`.
- `AI_TRADING_POSITION_SIZE_MIN_USD`: minimum per-trade notional in USD. Defaults to **$25** when unset. Values ≤0 are considered invalid and the risk engine automatically falls back to the default (guaranteeing at least one share when prices are high).

### Execution exposure tracking

- `ExecutionEngine.execute_order(...)` now returns an `ExecutionResult` object (a string subclass) containing the created order, its current status, the filled quantity, and the proportional signal weight that filled. Code that only needs the order ID can continue to treat the result as a string.
- The trading loop inspects the returned `ExecutionResult` and only calls `RiskEngine.register_fill(...)` when a non-zero fill quantity is reported. Partial fills are forwarded with a scaled signal weight so exposure reflects the filled portion of the order.
- The execution engine registers asynchronous callbacks with the order manager. Late fills (for example, fills confirmed after the trading cycle completes) automatically trigger `RiskEngine.register_fill(...)` with the outstanding fill delta, keeping the exposure ledger in sync even when confirmations arrive after the synchronous trading loop.
