from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from ai_trading.scheduler.aligned_clock import AlignedClock


class _NoTZCal:
    """Calendar stub without tz attribute."""


def test_default_to_utc_when_calendar_missing_tz(monkeypatch):
    clock = AlignedClock()
    clock.calendar = _NoTZCal()
    now = clock.get_exchange_time()
    assert now.tzinfo is UTC


def test_calendar_free_market_hours_compare_in_new_york():
    clock = AlignedClock()
    clock.calendar = None

    after_close_in_new_york = datetime(
        2024,
        1,
        3,
        10,
        0,
        tzinfo=ZoneInfo("Asia/Tokyo"),
    )
    regular_session_in_new_york = datetime(2024, 1, 2, 14, 45, tzinfo=UTC)

    assert clock.is_market_open("AAPL", after_close_in_new_york) is False
    assert clock.is_market_open("AAPL", regular_session_in_new_york) is True
