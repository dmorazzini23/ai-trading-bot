import pandas as pd
import pytest

from ai_trading.core import bot_engine


def _sample_df():
    return pd.DataFrame({"close": [1.0]}, index=[pd.Timestamp("2024-01-01", tz="UTC")])


def test_fetch_minute_df_safe_returns_dataframe(monkeypatch):
    monkeypatch.setattr(bot_engine, "get_minute_df", lambda s, start, end: _sample_df())
    monkeypatch.setattr(bot_engine, "_ensure_data_fresh", lambda symbols, max_age_seconds: None)
    result = bot_engine.fetch_minute_df_safe("AAPL")
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_fetch_minute_df_safe_raises_on_empty(monkeypatch):
    monkeypatch.setattr(bot_engine, "get_minute_df", lambda s, start, end: pd.DataFrame())
    monkeypatch.setattr(bot_engine, "_ensure_data_fresh", lambda symbols, max_age_seconds: None)
    with pytest.raises(bot_engine.DataFetchError):
        bot_engine.fetch_minute_df_safe("AAPL")
