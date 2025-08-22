import os
import sys
import time
from datetime import datetime, timezone
import pathlib

import pytest

# --- Keep tests in UTC ---
os.environ.setdefault("TZ", "UTC")
if hasattr(time, "tzset"):
    time.tzset()

# --- Seed dummy keys to avoid live calls ---
os.environ.setdefault("ALPACA_API_KEY", "test_key")
os.environ.setdefault("ALPACA_SECRET_KEY", "test_secret")

# --- Block network by default (opt-in via RUN_INTEGRATION=1) ---
from tests._netblock import block_network, should_block  # noqa: E402


def pytest_configure(config: pytest.Config) -> None:
    # Display UTC stamp at session start for debugging
    config._utc_stamp = datetime.now(timezone.utc).isoformat()  # noqa: SLF001
    config.addinivalue_line("markers", "integration: network/vendor tests")
    config.addinivalue_line("markers", "slow: long-running tests")
    pathlib.Path("artifacts").mkdir(exist_ok=True)
    pathlib.Path("artifacts/utc.txt").write_text(config._utc_stamp)
    if should_block():
        block_network()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if should_block():
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
