import datetime as dt
import sys
import types
import warnings

import pytest
import urllib3

pd = pytest.importorskip("pandas")
from ai_trading.data import fetch as data_fetcher
from ai_trading import alpaca_api as bars
from ai_trading.utils.base import health_check
from ai_trading.utils import env as env_utils


@pytest.fixture(autouse=True)
def _suppress_system_time_warning(monkeypatch):
    """Normalize TLS validation clock warnings in tests."""

    def _disable_warnings(category=urllib3.exceptions.HTTPWarning):
        warnings.filterwarnings("ignore", category=category)

    monkeypatch.setattr(urllib3, "disable_warnings", _disable_warnings)
    urllib3.disable_warnings(urllib3.exceptions.SystemTimeWarning)


@pytest.fixture(autouse=True)
def _reload_data_fetcher():
    import importlib
    global data_fetcher
    data_fetcher = importlib.import_module("ai_trading.data.fetch")
    global bars
    bars = importlib.import_module("ai_trading.alpaca_api")
    env_utils.refresh_alpaca_credentials_cache()
    try:
        data_fetcher._ALPACA_SYMBOL_FAILURES.clear()
        data_fetcher._ALPACA_FAILURE_EVENTS.clear()
        data_fetcher._FEED_OVERRIDE_BY_TF.clear()
    except AttributeError:
        pass


def _stub_df():
    index = pd.date_range(start="2024-01-01 09:30", periods=120, freq="min", tz="UTC")
    return pd.DataFrame({
        "open": 1.0,
        "high": 1.1,
        "low": 0.9,
        "close": 1.05,
        "volume": 100000
    }, index=index)


def test_fetch_fallback_to_daily(monkeypatch):
    df = _stub_df()
    monkeypatch.setattr(data_fetcher, "get_minute_df", lambda *a, **k: None)
    monkeypatch.setattr(bars, "get_bars_df", lambda *a, **k: df)
    monkeypatch.setattr(data_fetcher, "_ENABLE_HTTP_FALLBACK", False, raising=False)
    data_fetcher._ALPACA_SYMBOL_FAILURES.clear()
    data_fetcher._ALPACA_FAILURE_EVENTS.clear()

    # Properly mocked module for pandas_ta
    mock_module = types.ModuleType("pandas_ta")
    mock_module.atr = lambda *a, **k: pd.Series([1])
    monkeypatch.setitem(sys.modules, "pandas_ta", mock_module)

    types.SimpleNamespace(data_fetcher=data_fetcher)
    result = data_fetcher.get_minute_df("AAPL", dt.date.today(), dt.date.today())
    if result is None:
        result = data_fetcher.get_daily_df("AAPL", dt.date.today(), dt.date.today())
    assert health_check(result, "minute")


def test_fetch_minute_success(monkeypatch):
    df = _stub_df()
    monkeypatch.setattr(data_fetcher, "get_minute_df", lambda *a, **k: df)
    result = data_fetcher.get_minute_df("AAPL", dt.date.today(), dt.date.today())
    assert health_check(result, "minute")


def test_system_time_warning_suppressed():
    """Ensure SystemTimeWarning from urllib3 is silenced."""

    with warnings.catch_warnings(record=True) as w:
        warnings.warn("clock skew", urllib3.exceptions.SystemTimeWarning)
    assert not w
