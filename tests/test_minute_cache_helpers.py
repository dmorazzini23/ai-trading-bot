"""Test minute-bar cache helper functions."""

from datetime import UTC, datetime, timedelta

from ai_trading.data_fetcher import (
    clear_cached_minute_cache,
    get_cached_age_seconds,
    get_cached_minute_timestamp,
    set_cached_minute_timestamp,
)


def setup_function() -> None:
    clear_cached_minute_cache()


def test_set_and_get() -> None:
    assert get_cached_minute_timestamp("AAPL") is None
    ts = datetime.now(UTC)
    set_cached_minute_timestamp("AAPL", ts)
    assert get_cached_minute_timestamp("AAPL") == ts


def test_get_cached_age_seconds() -> None:
    now = datetime.now(UTC)
    earlier = now - timedelta(seconds=30)
    set_cached_minute_timestamp("MSFT", earlier)
    age = get_cached_age_seconds("MSFT", now=now)
    assert age is not None
    assert 29 <= age <= 31


def test_clear_cached_timestamp() -> None:
    ts = datetime.now(UTC)
    set_cached_minute_timestamp("AAPL", ts)
    clear_cached_minute_cache("AAPL")
    assert get_cached_minute_timestamp("AAPL") is None


def test_clear_all() -> None:
    ts = datetime.now(UTC)
    set_cached_minute_timestamp("AAPL", ts)
    set_cached_minute_timestamp("MSFT", ts)
    clear_cached_minute_cache()
    assert get_cached_minute_timestamp("AAPL") is None
    assert get_cached_minute_timestamp("MSFT") is None
