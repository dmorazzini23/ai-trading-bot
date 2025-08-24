"""
Alpaca vendor stub wiring fix (tests only)
-----------------------------------------
Ensure `alpaca_trade_api` and `alpaca_trade_api.rest` resolve to the
local test vendor stubs so imports like
    from alpaca_trade_api import REST
    from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
work during test collection. This is idempotent and does not affect runtime.
"""
import os
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")  # AI-AGENT-REF: disable plugin autoload
import sys as _sys

# AI-AGENT-REF: explicitly load xdist plugin when available
try:
    import importlib.util as _util
    _xdist_present = bool(_util.find_spec("xdist"))
except Exception:  # pragma: no cover - absence or lookup issue
    _xdist_present = False
pytest_plugins = ["xdist.plugin"] if _xdist_present else []

import importlib as _importlib

try:
    import alpaca_trade_api as _alpaca_mod  # may be real lib or existing stub
except Exception:  # pragma: no cover - only hit in test bootstrap
    _alpaca_mod = None

# If the module is absent or lacks expected attributes, bind to our stub package.
if _alpaca_mod is None or not hasattr(_alpaca_mod, "REST"):
    _alpaca_mod = _importlib.import_module("tests.vendor_stubs.alpaca_trade_api")
    _sys.modules["alpaca_trade_api"] = _alpaca_mod

# Ensure `alpaca_trade_api.rest` submodule provides TimeFrame/TimeFrameUnit
try:
    import alpaca_trade_api.rest as _alpaca_rest
except Exception:  # pragma: no cover
    _alpaca_rest = None

if (
    _alpaca_rest is None
    or not all(hasattr(_alpaca_rest, n) for n in ("TimeFrame", "TimeFrameUnit"))
):
    _alpaca_rest_stub = _importlib.import_module(
        "tests.vendor_stubs.alpaca_trade_api.rest"
    )
    _sys.modules["alpaca_trade_api.rest"] = _alpaca_rest_stub

import asyncio
import socket
import sys
from datetime import datetime, timezone
import pathlib

import pytest
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


def pytest_configure(config: pytest.Config) -> None:
    # Display UTC stamp at session start for debugging
    config._utc_stamp = datetime.now(timezone.utc).isoformat()  # noqa: SLF001
    config.addinivalue_line("markers", "integration: network/vendor tests")
    config.addinivalue_line("markers", "slow: long-running tests")
    config.addinivalue_line("markers", "legacy: legacy test quarantined during refactor")  # AI-AGENT-REF: register legacy marker
    pathlib.Path("artifacts").mkdir(exist_ok=True)
    pathlib.Path("artifacts/utc.txt").write_text(config._utc_stamp)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("RUN_INTEGRATION") not in {"1", "true", "TRUE", "yes"}:
        skip_integration = pytest.mark.skip(reason="integration tests require RUN_INTEGRATION=1")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


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

    _maybe_stub("alpaca_trade_api", lambda: _mk_pkg("alpaca_trade_api"))
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

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional for smoke
    pd = None  # AI-AGENT-REF: allow pandas absence for smoke


@pytest.fixture
def dummy_data_fetcher():
    if pd is None:
        pytest.skip("pandas required")  # AI-AGENT-REF: optional dep

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
    class DF:
        def get_minute_bars(self, symbol, start=None, end=None, limit=None):
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return DF()

