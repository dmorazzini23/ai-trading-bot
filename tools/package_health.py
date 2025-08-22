"""Quick health probes for CI diagnostics."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# AI-AGENT-REF: psutil probe for CI visibility

def _probe_psutil() -> bool:
    try:
        import psutil  # noqa: F401
        return True
    except Exception:
        return False


# AI-AGENT-REF: ensure Alpaca SDK presence is visible in CI
def _probe_alpaca_trade_api() -> bool:
    try:
        import alpaca_trade_api  # type: ignore
        getattr(alpaca_trade_api, "__version__", "unknown")
        return True
    except Exception:
        return False


def _probe_strategy_allocator() -> bool:
    try:
        from ai_trading.strategies.performance_allocator import (
            PerformanceBasedAllocator,
        )
        assert PerformanceBasedAllocator is not None
        return True
    except Exception:
        return False


# AI-AGENT-REF: check async test deps for CI visibility
def _probe_async_testing() -> bool:
    ok = True
    try:
        import pytest_asyncio  # type: ignore  # noqa: F401
    except Exception:
        ok = False
    try:
        import anyio  # type: ignore  # noqa: F401
    except Exception:
        ok = False
    return ok


def _probe_model_and_universe():
    import os
    os.getenv('AI_TRADER_TICKERS_FILE', 'tickers.csv')
    os.getenv('AI_TRADER_TICKERS_CSV')
    os.getenv('AI_TRADER_MODEL_PATH')
    os.getenv('AI_TRADER_MODEL_MODULE')
    try:
        import joblib  # noqa
    except Exception:
        pass


def _probe_model_config():
    import os

    p = os.getenv("AI_TRADER_MODEL_PATH")
    m = os.getenv("AI_TRADER_MODEL_MODULE")
    if p:
        pass
    elif m:
        pass
    else:
        return False
    try:
        import joblib  # noqa: F401
    except Exception:
        return False
    return True


if __name__ == "__main__":
    _probe_psutil()
    _probe_alpaca_trade_api()
    _probe_strategy_allocator()
    _probe_async_testing()
    _probe_model_and_universe()
    _probe_model_config()
