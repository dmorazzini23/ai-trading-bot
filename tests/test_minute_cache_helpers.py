from datetime import UTC, datetime

from ai_trading.data_fetcher import (
    _MINUTE_CACHE,  # type: ignore
    age_cached_minute_timestamps,
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


def test_age_cached_minute_timestamps() -> None:
    now = int(datetime.now(UTC).timestamp())
    earlier = now - 30
    set_cached_minute_timestamp("MSFT", earlier)
    # simulate old insertion
    _MINUTE_CACHE["MSFT"] = (earlier, now - 30)
    removed = age_cached_minute_timestamps(max_age_seconds=10)
    assert removed == 1
    assert get_cached_minute_timestamp("MSFT") is None


def test_clear_cached_timestamp() -> None:
    ts = int(datetime.now(UTC).timestamp())
    set_cached_minute_timestamp("AAPL", ts)
    clear_cached_minute_timestamp("AAPL")
    assert get_cached_minute_timestamp("AAPL") is None
