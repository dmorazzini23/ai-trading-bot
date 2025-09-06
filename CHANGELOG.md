# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### BREAKING CHANGES
- **Remove root import shims and fallbacks**
  - Deleted repo-root shims (`signals.py`, `data_fetcher.py`, `trade_execution.py`, `pipeline.py`, `indicators.py`, `portfolio.py`, `rebalancer.py`)
  - Removed all root-import fallbacks; use `ai_trading.*` imports exclusively
  - Added guard test preventing reintroduction of root imports
  - Updated README/CHANGELOG to document package-only policy

### Changed
- Config: validate Alpaca data feed entitlement and allow override via `ALPACA_DATA_FEED`, warning if switched.
- Data fetch: switch to SIP feed after first empty IEX result and record `data.fetch.feed_switch` metric.
- Main: finite `SCHEDULER_ITERATIONS` now exits promptly after completing
  the requested cycles instead of keeping the API thread alive. This
  avoids test/CI hangs; production runs continue to use infinite iterations.
- Makefile: add `PYTHON ?= python3` and route all invocations through
  `$(PYTHON)` for compatibility on Debian/Ubuntu where `python` shim is
  absent. Supports using a venv via `make ... PYTHON=.venv/bin/python`.

### Added
- Cache fallback data provider usage to skip redundant Alpaca requests
  for the same symbol and window.
- **Python**: restrict supported version to >=3.12,<3.13
- **Package Structure**: Root modules previously moved into `ai_trading/` package
  - **Migration Required**: Use `from ai_trading.signals import ...` instead of `from signals import ...`
  - **Breaking**: Root imports are no longer supported as of this version
- **Utils**: remove legacy `pathlib_shim` re-export; use `ai_trading.utils.paths` instead
- **Data Fetch**: validate Alpaca request parameters, check trading windows
  against the market calendar, retry up to 5 times, and optionally fall back to
  Yahoo when IEX returns empty.

### Fixed
- Dev deps: align `packaging` version with `constraints.txt` (25.0) to
  resolve resolver conflicts during `ensure-runtime` install.
- **Data Fetch**: raise error when configuration unavailable instead of repeated warnings.
- **Data Fetch**: improve Alpaca empty-bar handling by logging timeframe/feed,
  verifying API credentials and market hours, and attempting feed or window
  fallbacks before resorting to alternate providers.
- **Data Fetch**: retry with SIP feed when initial IEX request returns empty
  for a symbol that may be delisted or on the wrong feed.
- **Main**: `validate_environment` now raises `RuntimeError` when required
  environment variables are missing.
- Normalize broker-unavailable contract; remove false PDT warnings; add regression tests.
- Fix IndentationError in `bot_engine.py` (pybreaker stub); add static compile guard.
- **Runtime safety**: improved Alpaca availability checks, stable logging shutdown,
  lazy client init, config parity, and added utility/environment helpers.
- **Alpaca client**: log initialization failures, skip account logic when client
  unavailable, and abort trading loop if API remains unset.
- **ExecutionEngine**: Removed unsupported slippage metrics kwargs from initialization to prevent runtime `TypeError`.
- **Import Blocker**: Replaced corrupted `ai_trading/model_registry.py` with clean, minimal, typed model registry implementation
- Removed hard `data_client` dependency in risk engine with optional Alpaca client.
- Added `RLTrader` alias and completed config defaults for stable runtime.
- **BotContext**: expose portfolio weighting and rebalance attributes on `LazyBotContext` to avoid `AttributeError` during trades.
- Replaced test mock imports in sentiment module with local stub to avoid leakage.
  - Supports `register_model`, `load_model`, and `latest_for` operations
  - Maintains JSON index for model metadata
- **Logging**: handle `PermissionError` when creating log directories by warning and using secure permissions.
  - Includes dataset fingerprint verification for reproducibility
- **Configuration**: Corrected `DISABLE_DAILY_RETRAIN` environment flag parsing with safe default (`false`)
- **Import Hardening**: Made `model_pipeline` imports robust in `ai_trading/core/bot_engine.py`
  - Package-first import with graceful fallback to legacy root import
  - Works both as package import and when executed from repo root
- **Market hours**: cache closed-state logging to emit at most once per date.
- **Trading Loop**: guard missing Alpaca client and dedupe strategy logs
- **Alpaca API**: fix submit/retry logic including 429 handling
- **Staleness Guard**: convert timestamps to UTC with `pandas.to_datetime` and
  log naive vs aware conversions
- **Alpaca API**: validate `list_orders` availability and map alternative clients before trading loop
- **Alpaca API**: fix `list_orders` wrapper to forward `status` without introducing unsupported `statuses`
- **Config**: unify centralized defaults and add `from_optimization`
- **Utils**: re-export `ensure_utc` and enforce type assertions
- **validate_env**: support execution via `runpy.run_module`
- **Alpaca API**: provide lightweight fallback for `StockLatestQuoteRequest` to avoid startup ImportError when class is absent
- **CLI dry-run**: log indicator import confirmation and exit with code 0 before heavy imports.
- **Settings**: centralize value normalization and eliminate `FieldInfo` leaks
- **Position sizing**: fetch real account equity via Alpaca once and cache it to avoid repeated `EQUITY_MISSING` warnings.
- **Scheduler**: default to UTC when market calendar lacks timezone info.

### Added
- **Parallel Predictions**: Replaced single-threaded prediction executor with auto-sized thread pool
  - Default: `max(2, min(4, cpu_count))` workers to avoid thrashing
  - Environment override via `PREDICTION_WORKERS` environment variable
- **Data Staleness Checks**: Added minute-level data validation with UTC logging
  - `_ensure_data_fresh` function validates data age before signal evaluation
  - Configurable staleness threshold (default: 10 minutes during market hours)
  - All health check timestamps emitted in UTC for auditability

### Enhanced
- **Testing**: Added comprehensive test suite covering all changes
  - `test_model_registry.py`: Model registry round-trip validation
  - `test_config_env.py`: Environment flag parsing verification
  - `test_bot_engine_imports.py`: Import fallback logic testing
  - `test_prediction_executor.py`: Thread pool sizing validation
  - `test_staleness_guard.py`: Data freshness validation testing

### Migration Notes
- **Environment Variables**: New optional `PREDICTION_WORKERS` variable controls prediction thread pool size
- **Backward Compatibility**: All changes maintain existing API and behavior by default
- **Performance**: Prediction throughput should improve with parallel processing (2-4x typical improvement)
- **Monitoring**: Enhanced logging provides better visibility into data staleness and threading behavior
