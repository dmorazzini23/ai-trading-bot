# Migration Notes

## Monitoring API Unification & Cost-Aware Strategy Enhancement (Latest)

### Summary
Unified monitoring API to fix startup `ImportError` for `MetricsCollector`, replaced broad exception handling with specific exception types, and implemented cost-aware signal acceptance with ensemble gating for robust live trading.

### What Changed

**1. Monitoring API Unification**
- **Fixed:** `ImportError: MetricsCollector` from `ai_trading.monitoring.metrics`
- **Added:** Complete `MetricsCollector` class implementation in `ai_trading/monitoring/metrics.py`
- **Enhanced:** `PerformanceMonitor` class with unified interface combining metrics collection and performance analysis
- **Standardized:** All monitoring imports now use `PerformanceMonitor` as the primary monitoring interface
- **No shims:** All changes land in the package with single, canonical imports

**2. Exception Handling Hygiene**
- **Replaced:** Broad `except Exception:` patterns with specific exception types in production code
- **Enhanced:** Error logging with structured context (`component`, `error_type`, `symbol`)
- **Improved:** Entry points (`ai_trading/__main__.py`) with specific exception handling for:
  - `KeyboardInterrupt` → graceful exit
  - `ImportError`/`ModuleNotFoundError` → module import errors
  - `OSError`/`IOError` → I/O errors
  - Final `Exception` catch with structured logging
- **Updated:** Portfolio summary logging with specific exception types (`AttributeError`, `KeyError`, `ValueError`, `TypeError`)

**3. Cost-Aware Signal Enhancement**
- **Added:** `SignalDecisionPipeline` class for comprehensive signal evaluation
- **Implemented:** Cost-aware acceptance logic: reject signals where `expected_edge ≤ expected_transaction_cost + buffer`
- **Added:** Transaction cost estimation with commission, spread, and slippage modeling
- **Enhanced:** Decision pipeline with reason codes:
  - `ACCEPT_OK` → Signal passes all criteria
  - `REJECT_COST_UNPROFITABLE` → Costs exceed expected edge
  - `REJECT_EDGE_TOO_LOW` → Edge below minimum threshold
  - `REJECT_REGIME_HIGH_VOL` → High volatility regime with insufficient edge
  - `REJECT_ENSEMBLE_DISAGREEMENT` → Ensemble models disagree
  - `REJECT_DATA_ERROR` → Data quality issues
  - `REJECT_SYSTEM_ERROR` → System errors

**4. Strategy Logic Enhancements**
- **Added:** ATR-scaled exits with configurable stop/target multipliers
- **Implemented:** Market regime detection (trending vs ranging, volatility analysis)
- **Added:** Ensemble gating requiring N-of-M model agreement before trading
- **Enhanced:** Signal scoring with `score = predicted_edge - transaction_costs - buffer`
- **Added:** Per-signal logging with comprehensive metrics and context

**5. Walk-Forward Validation**
- **Created:** Enhanced WFA runner script (`scripts/run_wfa.py`)
- **Integrated:** Cost-aware strategy validation
- **Added:** Comprehensive performance grading system
- **Implemented:** Artifact generation for validation results

### Production Impact
- ✅ **Fixes startup failure:** No more `ImportError: MetricsCollector`
- ✅ **Improved observability:** Structured logging with reason codes and metrics
- ✅ **Enhanced resilience:** Specific exception handling prevents silent failures
- ✅ **Better risk management:** Cost-aware signal acceptance reduces unprofitable trades
- ✅ **Regime awareness:** Adaptive thresholds based on market conditions
- ✅ **Quality control:** Ensemble gating and validation harness for strategy changes

### Breaking Changes
**None** - All changes are additive or enhance existing functionality without breaking existing interfaces.

### Validation Steps
1. **Import smoke test:** `python scripts/smoke_imports.py` → validates monitoring imports
2. **Service restart:** `systemctl restart ai-trading.service` → no ImportError, clean startup logs
3. **Paper trading:** First cycle logs show `ACCEPT_OK`/`REJECT_*` reason codes with metrics
4. **Walk-forward validation:** `python scripts/run_wfa.py --dry-run` → validates setup

### Configuration
New signal pipeline configuration options:
```python
{
    "min_edge_threshold": 0.001,        # 0.1% minimum edge requirement
    "transaction_cost_buffer": 0.0005,  # 0.05% safety buffer
    "ensemble_min_agree": 2,            # Minimum models that must agree
    "ensemble_total": 3,                # Total ensemble models
    "atr_stop_multiplier": 2.0,         # Stop loss ATR multiplier
    "atr_target_multiplier": 3.0,       # Take profit ATR multiplier
    "regime_volatility_threshold": 0.02 # 2% volatility regime threshold
}
```

---

## Portfolio Optimizer & Transaction Costs Migration (Previous)

### Summary
Moved portfolio optimization and transaction cost modules from `scripts/` to the main `ai_trading/` package to fix production import errors.

### What Changed

**1. Portfolio Optimizer**
- **Moved from:** `scripts/portfolio_optimizer.py`
- **Moved to:** `ai_trading/portfolio/optimizer.py`
- **Updated:** `ai_trading/portfolio/__init__.py` to export `PortfolioDecision`, `PortfolioOptimizer`, `create_portfolio_optimizer`

**2. Transaction Cost Calculator**
- **Moved from:** `scripts/transaction_cost_calculator.py`
- **Moved to:** `ai_trading/execution/transaction_costs.py`
- **Updated:** `ai_trading/signals.py` to import from new location

**3. Backward Compatibility**
- Both `scripts/` files are now compatibility shims that re-export from new locations
- No breaking changes for existing code

**4. Bug Fixes**
- Fixed `ENHANCED_CONFIG_AVAILABLE` reference in transaction cost calculator
- Improved exception handling in signals.py portfolio decision path
- Added specific `ValueError`/`KeyError` handling with better error messages

### Production Impact
- ✅ Fixes `ImportError: PortfolioDecision` in production
- ✅ No production modules import from `scripts/*` anymore
- ✅ Better error reporting for portfolio decision rejections
- ✅ Maintains all existing functionality

### Testing
```bash
python scripts/smoke_imports.py  # Should show all green checkmarks
python -c "import ai_trading.signals"  # Should succeed
```

---

# Migration Notes: Import Guards Removal & TradingConfig Enhancement

This document summarizes the changes made to eliminate import guards, unify TradingConfig, and harden execution/risk paths for production reliability.

## New Environment Variables

The following new environment variables have been added to TradingConfig and can be set in your `.env` file:

### Core Trading Parameters
- `TRADING_MODE` (default: "paper") - Trading mode: "paper" or "live"
- `ALPACA_BASE_URL` (default: "https://paper-api.alpaca.markets") - Alpaca API base URL
- `SLEEP_INTERVAL` (default: 1.0) - Sleep interval between operations in seconds
- `MAX_RETRIES` (default: 3) - Maximum number of retry attempts
- `BACKOFF_FACTOR` (default: 2.0) - Exponential backoff factor for retries
- `MAX_BACKOFF_INTERVAL` (default: 60.0) - Maximum backoff interval in seconds
- `PCT` (default: 0.05) - Percentage threshold for various operations

### Model and Scheduler Configuration
- `MODEL_PATH` (default: None) - Path to ML model files
- `SCHEDULER_ITERATIONS` (default: 0) - Number of scheduler iterations (0 = infinite)
- `SCHEDULER_SLEEP_SECONDS` (default: 60) - Sleep interval between scheduler cycles
- `WINDOW` (default: 20) - Rolling window size for calculations
- `ENABLED` (default: True) - Global enable/disable flag

### Rate Limiter Configuration
- `CAPACITY` (default: 100) - Token bucket capacity for rate limiting
- `REFILL_RATE` (default: 10.0) - Token refill rate per second
- `QUEUE_TIMEOUT` (default: 30.0) - Queue timeout in seconds for rate-limited requests

## Breaking Changes

### 1. Import Guards Removed
All `try/except ImportError` blocks have been removed from:
- `ai_trading/rebalancer.py`
- `ai_trading/signals.py`
- `ai_trading/execution/live_trading.py`
- `ai_trading/utils/base.py`

**Impact**: Dependencies are now required at import time. Missing dependencies will cause immediate import errors instead of falling back to stub implementations.

### 2. Required Dependencies
The following packages are now hard dependencies:
- `pandas_market_calendars>=4.3`
- `alpaca-py>=0.42.0`
- `hmmlearn>=0.3.0`
- `psutil>=5.9.8`

**Action Required**: Update your environment to install these packages:
```bash
python -m pip install -U pip
pip install -e .
pip install alpaca-py hmmlearn psutil pandas_market_calendars
```

If opting for `alpaca-py`, update broker modules, tests, and docs accordingly.

### 3. StrategyAllocator Changes
The `_Stub` fallback in `ai_trading/core/bot_engine.py` has been removed.

**Impact**: If `StrategyAllocator` cannot be resolved from `ai_trading.strategy_allocator` or `scripts.strategy_allocator`, the system will raise a `RuntimeError` instead of using a no-op stub.

**Action Required**: Ensure that `scripts/strategy_allocator.py` exists and contains a valid `StrategyAllocator` class.

### 4. Feature Availability Flags
Feature availability flags have been replaced with configuration-based checks:
- `ENHANCED_FEATURES_AVAILABLE` → `settings.ENABLE_PORTFOLIO_FEATURES`
- `PORTFOLIO_FIRST_AVAILABLE` → `settings.ENABLE_PORTFOLIO_FEATURES`
- `PORTFOLIO_OPTIMIZATION_AVAILABLE` → `settings.ENABLE_PORTFOLIO_FEATURES`

**Action Required**: Set `ENABLE_PORTFOLIO_FEATURES=true` in your environment if you use advanced portfolio features.

### 5. Alpaca SDK Import Changes
Alpaca SDK imports are now at module top level in `ai_trading/execution/live_trading.py`:
- `from alpaca.trading.client import TradingClient`
- `from alpaca.trading.enums import OrderSide, TimeInForce`
- `from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest`

**Impact**: Missing Alpaca SDK will cause immediate import errors.

### 6. Tooling Updates
- Test dependency hints now expect the modern `alpaca-py` SDK (`import alpaca`).

## Enhanced Features

### 1. TradingConfig Enhancements
- All missing attributes now have proper defaults
- Environment variable overrides for all parameters
- Safe dictionary export with `to_dict(safe=True)` method that redacts secrets
- Comprehensive validation for new fields

### 2. Rate Limiter Integration
The `RateLimiter` constructor now accepts a `TradingConfig` instance:
```python
from ai_trading.config.management import TradingConfig
from ai_trading.integrations.rate_limit import RateLimiter

config = TradingConfig.from_env()
rate_limiter = RateLimiter(config=config)
```

### 3. Improved Error Handling
Import errors now provide clear guidance on missing dependencies and installation steps.

## Validation

Use the provided smoke test script to validate your environment:
```bash
cd /path/to/ai-trading-bot
PYTHONPATH=. python scripts/smoke_imports.py
```

This script will:
- Test all critical imports
- Verify TradingConfig functionality
- Check StrategyAllocator resolution
- Validate new attribute availability

## Rollback Strategy

If you need to rollback these changes:
1. Restore import guards by wrapping imports in `try/except ImportError` blocks
2. Revert TradingConfig to remove new attributes
3. Restore `_Stub` fallback in StrategyAllocator
4. Make dependencies optional again

However, this is **not recommended** as it reduces production reliability and error observability.

## Support

For issues related to this migration:
1. Check that all required dependencies are installed
2. Verify environment variables are set correctly
3. Run the smoke test script to identify specific issues
4. Check logs for specific import or configuration errors

The changes improve production reliability by failing fast on missing dependencies and providing comprehensive configuration management.
