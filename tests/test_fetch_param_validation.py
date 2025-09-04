from datetime import UTC, datetime, timedelta

import pytest

from ai_trading.data import fetch


def _trading_range():
    start = datetime(2024, 1, 2, tzinfo=UTC)
    end = start + timedelta(minutes=1)
    return start, end


def test_invalid_feed_raises():
    start, end = _trading_range()
    with pytest.raises(ValueError):
        fetch._fetch_bars("AAPL", start, end, "1Min", feed="bogus")


def test_invalid_adjustment_raises():
    start, end = _trading_range()
    with pytest.raises(ValueError):
        fetch._fetch_bars("AAPL", start, end, "1Min", adjustment="bad")


def test_window_without_trading_session_raises():
    start = datetime(2024, 1, 6, tzinfo=UTC)
    end = start + timedelta(days=1)
    with pytest.raises(ValueError):
        fetch._fetch_bars("AAPL", start, end, "1Min")
