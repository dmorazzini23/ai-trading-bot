import asyncio
import os
import socket
import sys
from datetime import datetime, timezone
import pathlib

import pytest
from freezegun import freeze_time


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

import pandas as pd


@pytest.fixture
def dummy_data_fetcher():
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
