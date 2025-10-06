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

`HEALTH_TICK_SECONDS` (alias `AI_TRADING_HEALTH_TICK_SECONDS`) controls how often
the scheduler emits health ticks. The default is **300 seconds**. Production
deployments should keep the interval at or above **30 seconds**; lower values
are accepted for tests, but the runtime clamps the cadence back to 30 seconds
and logs `HEALTH_TICK_INTERVAL_BELOW_RECOMMENDED` so operators know the
override is only intended for short-lived scenarios.

### Singleton guard

The service refuses to start when another healthy `ai-trading` API is already
bound to the configured port (`9001` by default). On startup it probes
`http://127.0.0.1:<port>/healthz`; if it responds with `service=ai-trading`,
the new process logs `API_PORT_HEALTHY_ELSEWHERE` and exits with
`EADDRINUSE`. Operators should stop the existing `ai-trading.service` (or
release the port) before relaunching to avoid running duplicate trading loops.

### Paths & default files
- Trade log location is controlled by `TRADE_LOG_PATH` (or the legacy `AI_TRADING_TRADE_LOG_PATH`). The packaged systemd unit pins it to `/home/aiuser/ai-trading-bot/logs/trades.jsonl` and creates that directory with `0700` permissions for `aiuser`.
- Without an override the bot prefers `/var/log/ai-trading-bot/trades.jsonl`; if that directory—or an explicit override—cannot be created or written it first falls back to `./logs/trades.jsonl` relative to the working directory before considering state-directory locations. When this happens the process logs `TRADE_LOG_FALLBACK_USER_STATE` and `TRADE_LOGGER_FALLBACK_ACTIVE` once per boot so operators can inspect the condition while the bot continues running.
- The application initializes this trade log on startup via `ai_trading.core.bot_engine.get_trade_logger()` and trade execution lazily creates it if missing. Custom deployments should call it once if they bypass the standard entrypoint.
- Startup verifies this trade log path is writable, logs `TRADE_LOG_PATH_READY` with the resolved location, and exits only when it cannot create any writable location (including the fallback search paths).
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

### Market data freshness tolerance

- `AI_TRADING_MINUTE_DATA_FRESHNESS_TOLERANCE_SECONDS` (alias `MINUTE_DATA_FRESHNESS_TOLERANCE_SECONDS`) bounds how old minute-bar data may be before trading halts. The default is **900 seconds (15 minutes)**, matching the data-validation guard in `ai_trading.data_validation.core`. Lowering the value tightens the freshness requirement; raising it allows the bot to keep trading when providers lag slightly. Updates take effect without code changes because `fetch_minute_df_safe` and Alpaca fallbacks read the setting at runtime.

### Alpaca market-data troubleshooting

1. **Confirm SDK pinning.** Production deployments must run `alpaca-trade-api==3.2.0`. Check the installed wheel with `pip show alpaca-trade-api` on the host and redeploy if the version differs.
2. **Verify market-data entitlements.** Use the service credentials to call the `/v2/stocks/{symbol}/bars` endpoint via `curl` or the `StockHistoricalDataClient`. A `403` or IEX-only data indicates the account has fallen back to the free tier; open a support ticket to restore SIP access when required.
3. **Inspect live feed metadata.** Run an interactive shell and call `ai_trading.data.fetch.get_minute_df("AAPL", start, end)` while the incident is active. The returned DataFrame exposes `df.attrs["source_label"]` (and related keys) so you can confirm which provider served the data when `IEX_MINUTE_DATA_STALE` triggers.
4. **Restore primary feed preference.** After Alpaca minute data is healthy, clear any forced fallback by reloading the service (or calling the admin task) so that `provider_monitor` observes fresh bars and stops preferring the backup provider.

### Provider safe-mode & halt handling

Repeated Alpaca feed failures (such as `UNAUTHORIZED_SIP` or consecutive `MINUTE_GAPS_*` events) now trigger provider safe-mode:

1. `AlertType.PROVIDER_OUTAGE` is emitted with `severity=CRITICAL`.
2. `halt.flag` (configurable via `HALT_FLAG_PATH`/`AI_TRADING_HALT_FLAG_PATH`) is written to disk so operators see the halt without reading logs.
3. `provider_monitor.disable("alpaca")` (and `_sip`) activates a backoff window.
4. The core engine skips signal evaluation, refuses fallback-priced orders, and the execution layer blocks submissions while the halt file is present or the provider remains disabled.

**Resume flow:** Clear the halt flag once Alpaca confirms SIP coverage has returned. After the cooldown expires `provider_monitor.record_success("alpaca")` (automatically called on healthy fetches) resets safe-mode state so the trading loop can resume.

### Execution exposure tracking

- `ExecutionEngine.execute_order(...)` now returns an `ExecutionResult` object (a string subclass) containing the created order, its current status, the filled quantity, and the proportional signal weight that filled. Code that only needs the order ID can continue to treat the result as a string.
- The trading loop inspects the returned `ExecutionResult` and only calls `RiskEngine.register_fill(...)` when a non-zero fill quantity is reported. Partial fills are forwarded with a scaled signal weight so exposure reflects the filled portion of the order.
- The execution engine registers asynchronous callbacks with the order manager. Late fills (for example, fills confirmed after the trading cycle completes) automatically trigger `RiskEngine.register_fill(...)` with the outstanding fill delta, keeping the exposure ledger in sync even when confirmations arrive after the synchronous trading loop.
