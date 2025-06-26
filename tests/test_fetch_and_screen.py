import types
import datetime as dt
import pandas as pd
import sys

import bot_engine as bot
import data_fetcher
from utils import health_check


def _stub_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [100000],
        },
        index=[pd.Timestamp("2024-01-01T09:30:00Z")],
    )


def test_fetch_fallback_to_daily(monkeypatch):
    """Minute fetch None triggers daily fallback and screening."""
    df = _stub_df()
    monkeypatch.setattr(data_fetcher, "get_minute_df", lambda *a, **k: None)
    monkeypatch.setattr(data_fetcher, "get_daily_df", lambda *a, **k: df)
    monkeypatch.setitem(sys.modules, "pandas_ta", types.SimpleNamespace(atr=lambda *a, **k: pd.Series([1])))
    ctx = types.SimpleNamespace(data_fetcher=data_fetcher)
    result = data_fetcher.get_minute_df("AAPL", dt.date.today(), dt.date.today())
    if result is None:
        result = data_fetcher.get_daily_df("AAPL", dt.date.today(), dt.date.today())
    assert health_check(result, "minute")
    assert bot.screen_universe(["AAPL"], ctx) == ["AAPL"]


def test_fetch_minute_success(monkeypatch):
    """Minute fetch success path."""
    df = _stub_df()
    monkeypatch.setattr(data_fetcher, "get_minute_df", lambda *a, **k: df)
    result = data_fetcher.get_minute_df("AAPL", dt.date.today(), dt.date.today())
    assert health_check(result, "minute")
