from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from ai_trading.data.fetch import ensure_datetime
from ai_trading.data.timeutils import nyse_session_utc
from ai_trading.data.timeutils import ensure_utc_datetime


def test_nyse_session_dst():
    # July 15, 2024 (DST)
    s, e = nyse_session_utc(date(2024, 7, 15))
    assert s.hour == 13 and s.tzinfo == UTC
    assert e.hour == 20 and e.tzinfo == UTC


def test_nyse_session_standard():
    # Jan 15, 2024 (Standard Time)
    s, e = nyse_session_utc(date(2024, 1, 15))
    assert s.hour == 14 and e.hour == 21  # 14:30–21:00Z


def test_mixed_naive_datetime_semantics_are_intentional():
    naive = datetime(2025, 8, 20, 9, 30)

    assert ensure_utc_datetime(naive) == naive.replace(tzinfo=UTC)
    assert ensure_datetime(naive) == naive.replace(tzinfo=ZoneInfo("America/New_York")).astimezone(UTC)
