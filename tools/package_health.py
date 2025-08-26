"""Quick health probes for CI diagnostics."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _probe_psutil() -> bool:
    try:
        import psutil
        return True
    except (KeyError, ValueError, TypeError):
        return False

def _probe_alpaca_trade_api() -> bool:
    try:
        import alpaca_trade_api
        getattr(alpaca_trade_api, '__version__', 'unknown')
        return True
    except (KeyError, ValueError, TypeError):
        return False

def _probe_strategy_allocator() -> bool:
    try:
        from ai_trading.strategies.performance_allocator import PerformanceBasedAllocator
        assert PerformanceBasedAllocator is not None
        return True
    except (KeyError, ValueError, TypeError):
        return False

def _probe_async_testing() -> bool:
    ok = True
    try:
        import pytest_asyncio
    except (KeyError, ValueError, TypeError):
        ok = False
    try:
        import anyio
    except (KeyError, ValueError, TypeError):
        ok = False
    return ok

def _probe_model_and_universe():
    import os
    os.getenv('AI_TRADING_TICKERS_FILE', os.getenv('AI_TRADER_TICKERS_FILE', 'tickers.csv'))
    os.getenv('AI_TRADING_TICKERS_CSV') or os.getenv('AI_TRADER_TICKERS_CSV')
    os.getenv('AI_TRADING_MODEL_PATH') or os.getenv('AI_TRADER_MODEL_PATH')
    os.getenv('AI_TRADING_MODEL_MODULE') or os.getenv('AI_TRADER_MODEL_MODULE')
    try:
        import joblib
    except (KeyError, ValueError, TypeError):
        pass

def _probe_model_config():
    import os
    p = os.getenv('AI_TRADING_MODEL_PATH') or os.getenv('AI_TRADER_MODEL_PATH')
    m = os.getenv('AI_TRADING_MODEL_MODULE') or os.getenv('AI_TRADER_MODEL_MODULE')
    if p:
        pass
    elif m:
        pass
    else:
        return False
    try:
        import joblib
    except (KeyError, ValueError, TypeError):
        return False
    return True
if __name__ == '__main__':
    _probe_psutil()
    _probe_alpaca_trade_api()
    _probe_strategy_allocator()
    _probe_async_testing()
    _probe_model_and_universe()
    _probe_model_config()