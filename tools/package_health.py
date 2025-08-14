"""Quick health probes for CI diagnostics."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# AI-AGENT-REF: psutil probe for CI visibility

def _probe_psutil() -> bool:
    try:
        import psutil  # noqa: F401
        print("[health] psutil: ok")
        return True
    except Exception as e:
        print("[health] psutil: MISSING ->", e)
        return False


# AI-AGENT-REF: ensure Alpaca SDK presence is visible in CI
def _probe_alpaca_trade_api() -> bool:
    try:
        import alpaca_trade_api  # type: ignore
        ver = getattr(alpaca_trade_api, "__version__", "unknown")
        print(f"[health] alpaca_trade_api: ok (version={ver})")
        return True
    except Exception as e:
        print("[health] alpaca_trade_api: MISSING ->", e)
        return False


def _probe_strategy_allocator() -> bool:
    try:
        import ai_trading.strategy_allocator as sa  # type: ignore
        assert hasattr(sa, "StrategyAllocator")
        print("[health] strategy_allocator: ok")
        return True
    except Exception as e:
        print("[health] strategy_allocator: MISSING/INVALID ->", e)
        return False


# AI-AGENT-REF: check async test deps for CI visibility
def _probe_async_testing() -> bool:
    ok = True
    try:
        import pytest_asyncio  # type: ignore  # noqa: F401
        print("[health] pytest-asyncio: ok")
    except Exception as e:
        ok = False
        print("[health] pytest-asyncio: MISSING ->", e)
    try:
        import anyio  # type: ignore  # noqa: F401
        print("[health] anyio: ok")
    except Exception as e:
        ok = False
        print("[health] anyio: MISSING ->", e)
    return ok


def _probe_model_config():
    import os

    p = os.getenv("AI_TRADER_MODEL_PATH")
    m = os.getenv("AI_TRADER_MODEL_MODULE")
    if p:
        print(f"[health] model: path -> {p}")
    elif m:
        print(f"[health] model: module -> {m}")
    else:
        print("[health] model: MISSING (required)")
        return False
    try:
        import joblib  # noqa: F401
        print("[health] joblib: ok")
    except Exception as e:
        print("[health] joblib: MISSING ->", e)
        return False
    return True


if __name__ == "__main__":
    _probe_psutil()
    _probe_alpaca_trade_api()
    _probe_strategy_allocator()
    _probe_async_testing()
    _probe_model_config()
