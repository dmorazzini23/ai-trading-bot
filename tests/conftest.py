"""
Alpaca vendor stub wiring fix (tests only)
-----------------------------------------
Ensure the `alpaca` package and its key submodules resolve to local vendor
stubs so imports like
    from alpaca.trading.client import TradingClient
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
work during test collection. Legacy `alpaca_trade_api` imports are aliased to
these stubs for backward compatibility. This is idempotent and does not affect
runtime.
"""
import os
import sys as _sys

import importlib as _importlib

try:
    from alpaca.trading.client import TradingClient  # type: ignore
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # type: ignore
except Exception:  # pragma: no cover - only hit in test bootstrap
    _alpaca_pkg = _importlib.import_module("tests.vendor_stubs.alpaca")
    _sys.modules.setdefault("alpaca", _alpaca_pkg)
    _sys.modules.setdefault(
        "alpaca.trading",
        _importlib.import_module("tests.vendor_stubs.alpaca.trading"),
    )
    _sys.modules.setdefault(
        "alpaca.trading.client",
        _importlib.import_module("tests.vendor_stubs.alpaca.trading.client"),
    )
    _sys.modules.setdefault(
        "alpaca.data",
        _importlib.import_module("tests.vendor_stubs.alpaca.data"),
    )
    _sys.modules.setdefault(
        "alpaca.data.timeframe",
        _importlib.import_module("tests.vendor_stubs.alpaca.data.timeframe"),
    )
    _sys.modules.setdefault(
        "alpaca.data.requests",
        _importlib.import_module("tests.vendor_stubs.alpaca.data.requests"),
    )
    from alpaca.trading.client import TradingClient  # noqa: F401
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # noqa: F401

# Alias legacy module names to new stubs for compatibility
_sys.modules.setdefault(
    "alpaca_trade_api",
    _importlib.import_module("tests.vendor_stubs.alpaca_trade_api"),
)
_sys.modules.setdefault(
    "alpaca_trade_api.rest",
    _importlib.import_module("tests.vendor_stubs.alpaca_trade_api.rest"),
)

import asyncio
import random
import socket
import sys
from datetime import datetime, timezone
import pathlib

import pytest
import types
try:
    # Optional dev dependency. Provide a benign fallback for smoke/collect.
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


def pytest_configure(config: pytest.Config) -> None:
    # Display UTC stamp at session start for debugging
    config._utc_stamp = datetime.now(timezone.utc).isoformat()  # noqa: SLF001
    config.addinivalue_line("markers", "integration: network/vendor tests")
    config.addinivalue_line("markers", "slow: long-running tests")
    pathlib.Path("artifacts").mkdir(exist_ok=True)
    pathlib.Path("artifacts/utc.txt").write_text(config._utc_stamp)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("RUN_INTEGRATION") not in {"1", "true", "TRUE", "yes"}:
        skip_integration = pytest.mark.skip(reason="integration tests require RUN_INTEGRATION=1")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


@pytest.fixture
def dummy_alpaca_client():
    class Client:
        def __init__(self):
            self.calls = 0

        def submit_order(self, **order_data):
            self.calls += 1
            return types.SimpleNamespace(id="123", **order_data)

    return Client()


def _install_vendor_stubs() -> None:
    import importlib
    import types

    def _maybe_stub(modname: str, creator):
        try:
            importlib.import_module(modname)
        except Exception:
            sys.modules[modname] = creator()

    def _mk_pkg(path: str) -> types.ModuleType:
        import pathlib as _pathlib
        pkg_name = path.replace("/", ".")
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(_pathlib.Path(__file__).parent / "vendor_stubs" / path)]
        return pkg

    _maybe_stub("alpaca", lambda: _mk_pkg("alpaca"))
    _maybe_stub("yfinance", lambda: importlib.import_module("tests.vendor_stubs.yfinance"))

_install_vendor_stubs()


# Minimal timing helpers for tests only (no package shim).
try:
    from ai_trading.utils import now, elapsed_ms  # preferred if it exists
except Exception:
    import time as _t

    def now() -> float:  # AI-AGENT-REF: fallback timer
        return _t.perf_counter()

    def elapsed_ms(start: float) -> float:  # AI-AGENT-REF: fallback timer
        return (_t.perf_counter() - start) * 1000.0

@pytest.fixture
def dummy_data_fetcher():
    pd = pytest.importorskip("pandas")

    class DF:
        def get_minute_bars(self, symbol, start=None, end=None, limit=None):
            idx = pd.date_range(end=datetime.now(timezone.utc), periods=30, freq="min")
            return pd.DataFrame(
                {"open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 1000},
                index=idx,
            )

    return DF()


@pytest.fixture
def dummy_data_fetcher_empty():
    pd = pytest.importorskip("pandas")

    class DF:
        def get_minute_bars(self, symbol, start=None, end=None, limit=None):
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    return DF()

