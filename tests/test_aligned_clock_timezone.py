from datetime import UTC

from ai_trading.scheduler.aligned_clock import AlignedClock


class _NoTZCal:
    """Calendar stub without tz attribute."""


def test_default_to_utc_when_calendar_missing_tz(monkeypatch):
    clock = AlignedClock()
    clock.calendar = _NoTZCal()
    now = clock.get_exchange_time()
    assert now.tzinfo is UTC
