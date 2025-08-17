"""Test minute-bar cache helper functions."""

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
    set_cached_minute_timestamp("AAPL", 100)
    assert get_cached_minute_timestamp("AAPL") == 100


def test_age_cached_timestamp() -> None:
    set_cached_minute_timestamp("MSFT", 200)
    assert age_cached_minute_timestamp("MSFT", 10) == 210
    assert age_cached_minute_timestamp("MSFT", -10) == 200


def test_clear_cached_timestamp() -> None:
    set_cached_minute_timestamp("AAPL", 123)
    clear_cached_minute_timestamp("AAPL")
    assert get_cached_minute_timestamp("AAPL") is None


def test_age_missing_symbol() -> None:
    assert age_cached_minute_timestamp("MIA", 5) is None
