# Recent Fixes Summary

## Pending order severity and cleanup cadence
- Pending orders now emit `PENDING_ORDERS_DETECTED` and `PENDING_ORDERS_STILL_PRESENT` at warning level, while successful cleanups log at info level so alerts match urgency.
- Tune or revert behaviour by adjusting `ORDER_STALE_CLEANUP_INTERVAL` (seconds) in the trading runtime config; increasing the interval delays cleanup and reduces warning frequency.

## Provider health decisions stay single-sourced
- The provider monitor now keeps a single active provider per pair, requiring two healthy passes plus cooldown expiry before switching back to primary, preventing oscillation.
- Tuning knobs: `DATA_COOLDOWN_SECONDS` (minimum healthy cooldown), `DATA_PROVIDER_BACKOFF_FACTOR`, and `DATA_PROVIDER_MAX_COOLDOWN` all live in the env-config layer. Lower the cooldown to hasten recovery or raise the backoff factor to be more conservative.

## Unified Alpaca credential resolution
- `resolve_alpaca_credentials` and `initialize` use the same dataclass wrapper, updating the cached credential state used by other modules.
- Override via `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, and `ALPACA_BASE_URL`. Clearing the keys (or setting `shadow=True`) reverts to stub behaviour.

## Yahoo intraday fallback reliability
- Oversized 1-minute windows now split into ≤8-day segments so Yahoo fallback remains reliable when Alpaca minute data is unavailable.
- Operators can bias the system back toward Alpaca by disabling fallback (`ENABLE_FINNHUB=1` with valid keys, or altering provider priority) or by shortening the requested window.

## Broker capacity preflight detection
- Capacity exhaustion responses from Alpaca raise `NonRetryableBrokerError`, incrementing capacity skip stats and preventing wasteful retries.
- Modify retry sensitivity by changing the execution engine's `retry_config` (e.g., lowering `max_attempts` or adjusting delays) in runtime overrides if you need to revert to permissive behaviour.

## Transient order retry hygiene
- Execution retries only repeat on transient network/API errors; non-retryable signals bubble immediately.
- Tune retry aggressiveness through deployment overrides of `ExecutionEngine.retry_config` or by wrapping the engine with a custom retry policy.

## Finnhub-disabled log deduplication
- The once-logger now keys on symbol, timeframe, and window so repeated fallback checks for different ranges still log once per unique window.
- To revert, disable the emit-once logger via logging configuration or force debug logging if every attempt must be recorded.

## Trade log cached exactly once
- `_load_trade_log_cache` hydrates the trade log only once per process, reusing the cached structure for later consumers.
- Force a refresh by clearing `_TRADE_LOG_CACHE_LOADED` or by disabling memoisation via config flag `META_SYNC_FROM_BROKER=false` if broker sync should always re-read.

## Daily fetch memo debounce
- Daily bar fetches memoize per-symbol results for `DAILY_FETCH_MEMO_TTL` seconds, avoiding duplicate requests when the schedule polls frequently.
- Shorten or disable the debounce by lowering `DAILY_FETCH_MEMO_TTL` (0 turns off memoisation); increase it to stretch the warm cache window.
