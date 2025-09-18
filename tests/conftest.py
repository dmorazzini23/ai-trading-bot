"""Test configuration and shared fixtures."""

import os
import sys as _sys

import asyncio
import random
import socket
import sys
from datetime import datetime, timezone
import pathlib
import types

import pytest
import ai_trading.data.fetch as data_fetcher


if "flask" not in _sys.modules:
    flask_mod = types.ModuleType("flask")
    flask_app_mod = types.ModuleType("flask.app")

    class _StubFlask:
        def __init__(self, *a, **k):
            self._routes = {}
            self.config = {}

        def route(self, path, *a, **k):
            def decorator(func):
                self._routes[path] = func
                return func

            return decorator

        def run(self, *a, **k):
            pass

        def test_client(self):
            app = self

            class _Response:
                def __init__(self, data, status=200):
                    self._data = data
                    self.status_code = status

                def get_json(self):
                    return self._data

            class _Client:
                def get(self, path):
                    handler = app._routes[path]
                    result = handler()
                    status = 200
                    data = result
                    if isinstance(result, tuple):
                        data = result[0]
                        if len(result) > 1:
                            status = result[1]
                    return _Response(data, status)

            return _Client()

    def _jsonify(payload=None, *args, **kwargs):
        if payload is not None:
            return payload
        if args:
            return args[0] if len(args) == 1 else list(args)
        return kwargs

    flask_mod.Flask = _StubFlask
    flask_mod.jsonify = getattr(flask_mod, "jsonify", _jsonify)
    flask_app_mod.Flask = _StubFlask
    flask_app_mod.jsonify = flask_mod.jsonify
    _sys.modules["flask"] = flask_mod
    _sys.modules["flask.app"] = flask_app_mod

try:
    from alpaca.trading.client import TradingClient  # type: ignore  # noqa: F401
    from alpaca.data import TimeFrame  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - dependency missing
    from tests.vendor_stubs import alpaca as _alpaca
    import sys as _sys

    _sys.modules.setdefault("alpaca", _alpaca)
    _sys.modules.setdefault("alpaca.trading", _alpaca.trading)
    _sys.modules.setdefault("alpaca.trading.client", _alpaca.trading.client)
    _sys.modules.setdefault("alpaca.data", _alpaca.data)
    _sys.modules.setdefault("alpaca.data.timeframe", _alpaca.data.timeframe)
    _sys.modules.setdefault("alpaca.data.requests", _alpaca.data.requests)

    from tests.vendor_stubs.alpaca.trading.client import TradingClient  # type: ignore  # noqa: F401
    from tests.vendor_stubs.alpaca.data.timeframe import TimeFrame  # type: ignore  # noqa: F401

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


@pytest.fixture(autouse=True)
def _reset_fallback_cache(monkeypatch):
    monkeypatch.setattr(data_fetcher, "_FALLBACK_WINDOWS", set())
    monkeypatch.setattr(data_fetcher, "_FALLBACK_UNTIL", {})


@pytest.fixture
def dummy_order():
    """Provide a minimal order-like object for tests.

    Ensures required attributes are present and ``filled_qty`` defaults to ``0``
    so that code paths handling numeric comparisons do not encounter ``None``.
    """

    return types.SimpleNamespace(
        id="1",
        status="filled",
        filled_qty=0,
        symbol="TEST",
    )
