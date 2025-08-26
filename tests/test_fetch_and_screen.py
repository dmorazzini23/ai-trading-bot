import datetime as dt
import sys
import types

import pytest

pd = pytest.importorskip("pandas")
from ai_trading import data_fetcher
from ai_trading.utils.base import health_check


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
    monkeypatch.setattr(data_fetcher, "get_daily_df", lambda *a, **k: df)

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
