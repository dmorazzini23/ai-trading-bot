from datetime import date
from zoneinfo import ZoneInfo

from ai_trading.data.timeutils import nyse_session_utc


def test_nyse_session_dst():
    # July 15, 2024 (DST)
    s, e = nyse_session_utc(date(2024, 7, 15))
    assert s.hour == 13 and s.tzinfo == ZoneInfo("UTC")
    assert e.hour == 20 and e.tzinfo == ZoneInfo("UTC")


def test_nyse_session_standard():
    # Jan 15, 2024 (Standard Time)
    s, e = nyse_session_utc(date(2024, 1, 15))
    assert s.hour == 14 and e.hour == 21  # 14:30–21:00Z
