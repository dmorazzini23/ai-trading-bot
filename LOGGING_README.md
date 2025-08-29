## Logging Configuration

 The AI trading bot supports several logging configuration flags to control output verbosity and format. Logger initialization is idempotent—calling the setup multiple times will not create duplicate handlers. The setup routine now inspects existing handler *types* and ensures only one of each type is active, preventing accidental handler growth across repeated calls. A re-entrant lock guards the setup sequence so imports that indirectly invoke logging cannot deadlock during application startup.

### Environment Variables

- **`LOG_COMPACT_JSON`** (default: `false`): When enabled, uses compact JSON formatting that drops large extra payloads to prevent log truncation in terminals and journald.

- **`LOG_MARKET_FETCH`** (default: `false`): When enabled, logs periodic market fetch heartbeats at INFO level. When disabled, these messages are demoted to DEBUG level to reduce noise.

- **`LOG_LEVEL_YFINANCE`** (default: `WARNING`): Controls the log level for the `yfinance` package. Set to `INFO` or another level to troubleshoot provider interactions.

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
`***REDACTED***`.

#### Data Fetch Diagnostics
Daily price requests now log their parameters and outcome:

- `DAILY_FETCH_REQUEST` records the symbol, timeframe, and date range
- `DAILY_FETCH_RESULT` reports the number of rows returned and whether the
  response came from cache
- Empty bar responses log a warning only when the market is open; otherwise,
  fallback attempts proceed without additional noise

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
