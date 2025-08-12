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
- `alpaca-trade-api>=3.1.0`
- `hmmlearn>=0.3.0`
- `psutil>=5.9.8`

**Action Required**: Update your environment to install these packages:
```bash
pip install pandas_market_calendars alpaca-py alpaca-trade-api hmmlearn psutil
```

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