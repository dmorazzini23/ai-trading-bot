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


def test_window_without_trading_session_returns_empty():
    start = datetime(2024, 1, 6, tzinfo=UTC)
    end = start + timedelta(days=1)
    out = fetch._fetch_bars("AAPL", start, end, "1Min")
    assert out is None or out.empty


def test_missing_session_raises(monkeypatch):
    start = datetime(2024, 1, 2, 15, tzinfo=UTC)
    end = start + timedelta(minutes=1)
    monkeypatch.setattr(fetch, "_HTTP_SESSION", None, raising=False)
    with pytest.raises(ValueError):
        fetch._fetch_bars("AAPL", start, end, "1Min")


def test_missing_start_raises():
    _, end = _trading_range()
    with pytest.raises(ValueError):
        fetch._fetch_bars("AAPL", None, end, "1Min")


def test_missing_end_raises():
    start, _ = _trading_range()
    with pytest.raises(ValueError):
        fetch._fetch_bars("AAPL", start, None, "1Min")


def test_fetch_daily_async_requires_start():
    _, end = _trading_range()
    with pytest.raises(ValueError):
        fetch.fetch_daily_data_async(["AAPL"], None, end)


def test_fetch_daily_async_requires_end():
    start, _ = _trading_range()
    with pytest.raises(ValueError):
        fetch.fetch_daily_data_async(["AAPL"], start, None)


def test_sip_fallback_requires_session():
    with pytest.raises(ValueError):
        fetch._sip_fallback_allowed(None, {}, "1Min")
