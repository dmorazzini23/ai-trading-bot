import pytest
from datetime import datetime, UTC
from ai_trading.utils.lazy_imports import load_pandas

from ai_trading.core import bot_engine
from ai_trading.guards import staleness


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):  # noqa: D401
        return datetime(2024, 1, 1, 12, 5, 30, tzinfo=UTC)


def test_fetch_minute_df_safe_filters_current_and_zero_volume(monkeypatch):
    pd = load_pandas()
    idx = pd.to_datetime(
        [
            "2024-01-01 12:03:00+00:00",
            "2024-01-01 12:04:00+00:00",
            "2024-01-01 12:05:00+00:00",
        ]
    )
    df = pd.DataFrame(
        {
            "open": [1, 1, 1],
            "high": [1, 1, 1],
            "low": [1, 1, 1],
            "close": [1, 1, 1],
            "volume": [100, 0, 100],
            "timestamp": idx,
        },
        index=idx,
    )
    monkeypatch.setattr(bot_engine, "datetime", _FixedDatetime)
    monkeypatch.setattr(bot_engine, "get_minute_df", lambda s, start, end: df)
    monkeypatch.setattr(
        staleness,
        "_ensure_data_fresh",
        lambda df, max_age_seconds, *, symbol=None, now=None, tz=None: None,
    )

    result = bot_engine.fetch_minute_df_safe("AAPL")
    assert list(result.index) == [pd.Timestamp("2024-01-01 12:03:00+00:00")]
    assert (result["volume"] > 0).all()

