from __future__ import annotations

# Tests for NYSE market calendar wrapper.  # AI-AGENT-REF
from datetime import date

import pytest
from tests.optdeps import require
from ai_trading.data.market_calendar import is_trading_day, rth_session_utc

pmc = require("pandas_market_calendars")


def test_rth_dst_summer_standard_times() -> None:
    # Summer (DST): 09:30 ET -> 13:30 UTC
    s, e = rth_session_utc(date(2025, 8, 20))
    assert s.hour == 13 and s.minute == 30
    assert e.hour == 20 and e.minute == 0

    # Winter (standard): 09:30 ET -> 14:30 UTC
    s2, e2 = rth_session_utc(date(2025, 1, 6))
    assert s2.hour == 14 and s2.minute == 30
    assert e2.hour == 21 and e2.minute == 0


def test_known_early_close_black_friday() -> None:
    # Black Friday 2024-11-29: early close ~13:00 ET
    s, e = rth_session_utc(date(2024, 11, 29))
    assert s.hour == 14 and s.minute == 30  # 09:30 ET -> 14:30 UTC
    assert e.hour == 18 and e.minute == 0   # 13:00 ET -> 18:00 UTC


def test_is_trading_day_true_black_friday() -> None:
    assert is_trading_day(date(2024, 11, 29)) is True

