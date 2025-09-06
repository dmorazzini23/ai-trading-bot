## Logging Configuration

 The AI trading bot supports several logging configuration flags to control output verbosity and format. Logger initialization is idempotent—calling the setup multiple times will not create duplicate handlers. The setup routine now inspects existing handler *types* and ensures only one of each type is active, preventing accidental handler growth across repeated calls. A re-entrant lock guards the setup sequence so imports that indirectly invoke logging cannot deadlock during application startup.

### Environment Variables

- **`LOG_COMPACT_JSON`** (default: `false`): When enabled, uses compact JSON formatting that drops large extra payloads to prevent log truncation in terminals and journald.

- **`LOG_MARKET_FETCH`** (default: `false`): When enabled, logs periodic market fetch heartbeats at INFO level. When disabled, these messages are demoted to DEBUG level to reduce noise.

- **`LOG_LEVEL_YFINANCE`** (default: `WARNING`): Controls the log level for the `yfinance` package. Set to `INFO` or another level to troubleshoot provider interactions.
- **`LOG_QUIET_LIBRARIES`** (default: `charset_normalizer=INFO`): Comma-separated `logger=LEVEL` pairs used to suppress noisy third-party libraries.
- **`AI_TRADING_WARN_IF_MODEL_MISSING`** (default: `false`): When set, emits an `ML_MODEL_MISSING` warning if a configured model path cannot be found.

### Features

#### Emit-Once Logging
Startup banners and environment setup messages use emit-once logging to prevent duplicate messages during multiple initialization attempts:

- "Alpaca SDK is available" (or a warning if the SDK is missing)
- "FinBERT loaded successfully"
- Configuration verification messages

#### Compact JSON Format
When `LOG_COMPACT_JSON=true`, logs use a more compact format that:
- Omits non-essential extra fields
- Uses compact JSON serialization (no spaces)
- Prevents ellipsis truncation in typical terminals

#### Market Fetch Gating
The periodic `MARKET_FETCH` heartbeat logs can be controlled:
- `LOG_MARKET_FETCH=true`: Logs at INFO level (visible by default)
- `LOG_MARKET_FETCH=false`: Logs at DEBUG level (requires debug logging to see)

#### Risk Engine Quieting
Risk exposure update failures are demoted to DEBUG level when the trading context is not ready, reducing startup noise.

#### Automatic Secret Redaction
Any `extra` fields such as API keys, secrets, or URLs are sanitized before
being emitted to handlers, ensuring sensitive values are replaced with
`***REDACTED***`.  The :func:`ai_trading.logging.redact.redact_env` helper now
accepts ``drop=True`` to **remove** known sensitive keys instead of masking
them.  The early environment validation step uses this mode so that logs do
not contain secret key names or placeholder values.

#### Data Fetch Diagnostics
Daily price requests now log their parameters and outcome:

- `DAILY_FETCH_REQUEST` records the symbol, timeframe, and date range only when a cache miss triggers a fetch
- `DAILY_FETCH_CACHE_HIT` emits once on first cache use; subsequent hits appear only when debug logging is enabled
- `DAILY_FETCH_RESULT` reports the number of rows returned and whether the
  response came from cache
- Unauthorized feed responses trigger a quick entitlement check and switch to a
  permitted feed when available. If no alternative exists, operators are
  notified once.
- Empty bar responses are retried with an exponential backoff. The retry
  policy can be tuned via `FETCH_BARS_MAX_RETRIES`, `FETCH_BARS_BACKOFF_BASE`,
  and `FETCH_BARS_BACKOFF_CAP`. Logs include the remaining retry count,
  emit `ALPACA_FETCH_ABORTED` when a request is terminated early despite
  remaining retries, and emit `ALPACA_FETCH_RETRY_LIMIT` when no more attempts
  remain, allowing operators to diagnose persistent data issues quickly.

#### Backup Provider Usage
When the primary data source fails and the backup provider serves a window,
the system logs `BACKUP_PROVIDER_USED` and increments the
`backup_provider_used_total` Prometheus counter. The counter is labeled with
`provider` and `symbol`, allowing operators to track fallback frequency per
feed and instrument. Query the metric via the `/metrics` endpoint:

```bash
curl -sf http://localhost:$HEALTHCHECK_PORT/metrics | grep backup_provider_used_total
```

### Example Usage

```bash
# Enable compact logging for production
export LOG_COMPACT_JSON=true

# Hide market fetch heartbeats
export LOG_MARKET_FETCH=false

# Start the bot
python -m ai_trading.core.bot_engine
```

### Risk & Rollback

- Low risk changes (log-level modifications only)
- Toggle flags to revert behavior: `LOG_COMPACT_JSON=false`, `LOG_MARKET_FETCH=true`
- Revert to previous commit if needed
