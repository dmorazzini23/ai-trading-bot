import pytest
from datetime import UTC, datetime

from ai_trading.utils.lazy_imports import load_pandas

from ai_trading.core import bot_engine
from ai_trading.data import market_calendar
from ai_trading.guards import staleness
from ai_trading.utils import base as base_utils


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):  # noqa: D401
        return datetime(2024, 1, 1, 12, 5, 30, tzinfo=UTC)


def test_fetch_minute_df_safe_keeps_zero_volume_for_backup_provider(monkeypatch):
    pd = load_pandas()
    idx = pd.to_datetime(
        [
            "2024-01-01 12:00:00+00:00",
            "2024-01-01 12:01:00+00:00",
            "2024-01-01 12:02:00+00:00",
            "2024-01-01 12:03:00+00:00",
            "2024-01-01 12:04:00+00:00",
            "2024-01-01 12:05:00+00:00",
        ]
    )
    df = pd.DataFrame(
        {
            "open": [1, 1, 1, 1, 1, 1],
            "high": [1, 1, 1, 1, 1, 1],
            "low": [1, 1, 1, 1, 1, 1],
            "close": [1, 1, 1, 1, 1, 1],
            "volume": [100, 0, 100, -5, 150, 200],
            "timestamp": idx,
        },
        index=idx,
    )
    df.attrs["data_provider"] = "yahoo"
    df.attrs["fallback_provider"] = "yahoo"
    monkeypatch.setattr(bot_engine, "datetime", _FixedDatetime)
    monkeypatch.setattr(bot_engine, "_LONGEST_INTRADAY_INDICATOR_MINUTES", 1, raising=False)
    monkeypatch.setattr(bot_engine.CFG, "intraday_lookback_minutes", 1, raising=False)
    monkeypatch.setattr(bot_engine.CFG, "longest_intraday_indicator_minutes", 1, raising=False)
    monkeypatch.setattr(bot_engine.CFG, "intraday_indicator_window_minutes", 1, raising=False)
    monkeypatch.setattr(bot_engine.CFG, "market_cache_enabled", False, raising=False)
    monkeypatch.setattr(bot_engine.CFG, "data_feed", "iex", raising=False)
    monkeypatch.setattr(bot_engine.CFG, "minute_gap_backfill", "none", raising=False)
    monkeypatch.setattr(bot_engine, "get_minute_df", lambda s, start, end, **_: df)
    monkeypatch.setattr(base_utils, "is_market_open", lambda: True)
    monkeypatch.setattr(
        market_calendar,
        "rth_session_utc",
        lambda *_: (
            datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            datetime(2024, 1, 1, 12, 10, tzinfo=UTC),
        ),
    )
    monkeypatch.setattr(
        market_calendar,
        "previous_trading_session",
        lambda current_date: current_date,
    )
    monkeypatch.setattr(
        staleness,
        "_ensure_data_fresh",
        lambda df, max_age_seconds, *, symbol=None, now=None, tz=None: None,
    )

    result = bot_engine.fetch_minute_df_safe("AAPL")
    assert pd.Timestamp("2024-01-01 12:05:00+00:00") not in result.index
    assert pd.Timestamp("2024-01-01 12:03:00+00:00") not in result.index
    assert result.loc[pd.Timestamp("2024-01-01 12:01:00+00:00"), "volume"] == 0


def test_fetch_minute_df_safe_drops_string_nan_rows(monkeypatch):
    pd = load_pandas()
    idx = pd.to_datetime(
        [
            "2024-01-01 12:02:00+00:00",
            "2024-01-01 12:03:00+00:00",
            "2024-01-01 12:04:00+00:00",
        ]
    )
    df = pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2],
            "high": [1.0, 1.1, 1.2],
            "low": [1.0, 1.1, 1.2],
            "close": ["nan", 1.1, 1.2],
            "volume": [100, 150, 200],
            "timestamp": idx,
        },
        index=idx,
    )
    monkeypatch.setattr(bot_engine, "datetime", _FixedDatetime)
    monkeypatch.setattr(bot_engine, "_LONGEST_INTRADAY_INDICATOR_MINUTES", 1, raising=False)
    monkeypatch.setattr(bot_engine.CFG, "intraday_lookback_minutes", 1, raising=False)
    monkeypatch.setattr(bot_engine.CFG, "longest_intraday_indicator_minutes", 1, raising=False)
    monkeypatch.setattr(bot_engine.CFG, "intraday_indicator_window_minutes", 1, raising=False)
    monkeypatch.setattr(bot_engine.CFG, "market_cache_enabled", False, raising=False)
    monkeypatch.setattr(bot_engine.CFG, "data_feed", "iex", raising=False)
    monkeypatch.setattr(bot_engine.CFG, "minute_gap_backfill", "none", raising=False)
    monkeypatch.setattr(bot_engine, "get_minute_df", lambda s, start, end, **_: df)
    monkeypatch.setattr(base_utils, "is_market_open", lambda: True)
    monkeypatch.setattr(
        market_calendar,
        "rth_session_utc",
        lambda *_: (
            datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            datetime(2024, 1, 1, 12, 10, tzinfo=UTC),
        ),
    )
    monkeypatch.setattr(
        market_calendar,
        "previous_trading_session",
        lambda current_date: current_date,
    )
    monkeypatch.setattr(
        staleness,
        "_ensure_data_fresh",
        lambda df, max_age_seconds, *, symbol=None, now=None, tz=None: None,
    )

    result = bot_engine.fetch_minute_df_safe("AAPL")
    assert list(result.index) == [
        pd.Timestamp("2024-01-01 12:03:00+00:00"),
        pd.Timestamp("2024-01-01 12:04:00+00:00"),
    ]
    assert result["close"].tolist() == [1.1, 1.2]
    assert result["close"].isna().sum() == 0
    fast_ema = result["close"].ewm(span=12, min_periods=1, adjust=False).mean()
    slow_ema = result["close"].ewm(span=26, min_periods=1, adjust=False).mean()
    assert not fast_ema.dropna().empty
    assert not slow_ema.dropna().empty

