"""Test configuration and shared fixtures."""

import os
import sys as _sys

import asyncio
import random
import socket
import sys
from datetime import datetime, timezone
import pathlib

import pytest
import ai_trading.data.fetch as data_fetcher

try:
    from alpaca.trading.client import TradingClient  # type: ignore  # noqa: F401
    from alpaca.data import TimeFrame  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - dependency missing
    pytest.skip("alpaca-py is required for tests", allow_module_level=True)

try:
    from freezegun import freeze_time  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    from contextlib import contextmanager

    @contextmanager
    def freeze_time(*_a, **_k):  # AI-AGENT-REF: no-op when freezegun missing
        yield


@pytest.fixture(autouse=True, scope="session")
def _event_loop_policy():
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())


@pytest.fixture(autouse=True)
def _freeze_clock():
    with freeze_time("2024-01-02 15:04:05", tz_offset=0):
        yield


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    def guard(*a, **k):
        raise RuntimeError("Network disabled in tests")

    monkeypatch.setattr(socket, "create_connection", guard, raising=True)


@pytest.fixture(autouse=True)
def _env_defaults(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "dummy")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "dummy")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.setenv("CAPITAL_CAP", "0.04")
    monkeypatch.setenv("DOLLAR_RISK_LIMIT", "0.05")
    monkeypatch.setenv("MAX_POSITION_SIZE", "1000")


@pytest.fixture(autouse=True)
def _seed_prng() -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    random.seed(0)
    try:
        import numpy as np

        np.random.seed(0)
    except Exception:
        pass
    try:
        import torch  # type: ignore
    except Exception:
        pass
    else:
        torch.manual_seed(0)


def reload_module(mod):
    """Reload a module by object or name for tests."""
    import importlib

    if isinstance(mod, str):
        return importlib.reload(importlib.import_module(mod))
    return importlib.reload(mod)

# Compatibility fixtures expected by some tests
@pytest.fixture(name="default_env")
def _default_env(_env_defaults):  # reuse existing autouse env defaults
    """Alias fixture: provides default env via autouse `_env_defaults`."""
    yield

@pytest.fixture
def dummy_data_fetcher():
    """Provide a minimal data_fetcher interface for unit tests.

    Exposes `get_daily_df(ctx, sym)` and `get_minute_bars(sym)` returning
    a simple 30-row OHLCV DataFrame for deterministic checks.
    """
    import pandas as pd

    class _F:
        def __init__(self):
            n = 30
            self._df = pd.DataFrame(
                {
                    "open": [1.0] * n,
                    "high": [1.0] * n,
                    "low": [1.0] * n,
                    "close": [1.0] * n,
                    "volume": [100] * n,
                }
            )

        def get_daily_df(self, ctx, sym):  # noqa: ARG002 - ctx unused in tests
            return self._df.copy()

        def get_minute_bars(self, sym):  # noqa: ARG002
            return self._df.copy()

    return _F()


# Sanitize specific executor env vars for tests that perform naive int(os.getenv(...))
@pytest.fixture(autouse=True)
def _sanitize_executor_env(monkeypatch):
    import os as _os

    _orig_getenv = _os.getenv

    def _sanitized_getenv(key, default=None):  # type: ignore[override]
        if str(key).upper() in {"EXECUTOR_WORKERS", "PREDICTION_WORKERS"}:
            val = _orig_getenv(key, default)
            try:
                return val if (val is None or str(val).isdigit()) else ""
            except Exception:
                return ""
        return _orig_getenv(key, default)

    monkeypatch.setattr(_os, "getenv", _sanitized_getenv, raising=True)


@pytest.fixture(autouse=True)
def _reset_fallback_cache(monkeypatch):
    monkeypatch.setattr(data_fetcher, "_FALLBACK_WINDOWS", set())
