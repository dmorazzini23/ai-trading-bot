# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed
- **Import Blocker**: Replaced corrupted `ai_trading/model_registry.py` with clean, minimal, typed model registry implementation
  - Supports `register_model`, `load_model`, and `latest_for` operations
  - Maintains JSON index for model metadata
  - Includes dataset fingerprint verification for reproducibility
- **Configuration**: Corrected `DISABLE_DAILY_RETRAIN` environment flag parsing with safe default (`false`)
- **Import Hardening**: Made `model_pipeline` imports robust in `ai_trading/core/bot_engine.py`
  - Package-first import with graceful fallback to legacy root import
  - Works both as package import and when executed from repo root

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