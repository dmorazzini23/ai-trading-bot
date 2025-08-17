from datetime import UTC, datetime

from ai_trading.data_fetcher import (
    age_cached_minute_timestamp,
    clear_cached_minute_timestamp,
    get_cached_minute_timestamp,
    set_cached_minute_timestamp,
)


def setup_function() -> None:
    clear_cached_minute_timestamp("AAPL")
    clear_cached_minute_timestamp("MSFT")


def test_set_and_get() -> None:
    assert get_cached_minute_timestamp("AAPL") is None
    ts = int(datetime.now(UTC).timestamp())
    set_cached_minute_timestamp("AAPL", ts)
    assert get_cached_minute_timestamp("AAPL") == ts


def test_age_cached_minute_timestamp() -> None:
    now = int(datetime.now(UTC).timestamp())
    earlier = now - 30
    set_cached_minute_timestamp("MSFT", earlier)
    age = age_cached_minute_timestamp("MSFT", now_ts=now)
    assert age is not None
    assert 29 <= age <= 31


def test_clear_cached_timestamp() -> None:
    ts = int(datetime.now(UTC).timestamp())
    set_cached_minute_timestamp("AAPL", ts)
    clear_cached_minute_timestamp("AAPL")
    assert get_cached_minute_timestamp("AAPL") is None
