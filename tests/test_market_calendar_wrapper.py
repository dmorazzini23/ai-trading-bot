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


@pytest.mark.parametrize(
    "d, start_h, start_m, end_h, end_m",
    [
        (date(2024, 11, 29), 14, 30, 18, 0),
        (date(2025, 11, 28), 14, 30, 18, 0),
    ],
)
def test_known_early_close_black_friday(
    d: date, start_h: int, start_m: int, end_h: int, end_m: int
) -> None:
    s, e = rth_session_utc(d)
    assert s.hour == start_h and s.minute == start_m
    assert e.hour == end_h and e.minute == end_m


@pytest.mark.parametrize("d", [date(2024, 11, 29), date(2025, 11, 28)])
def test_is_trading_day_true_black_friday(d: date) -> None:
    assert is_trading_day(d) is True


def test_dst_transition_sessions() -> None:
    cases = [
        (date(2024, 3, 11), 13, 30, 20, 0),
        (date(2024, 11, 4), 14, 30, 21, 0),
        (date(2025, 3, 10), 13, 30, 20, 0),
        (date(2025, 11, 3), 14, 30, 21, 0),
    ]
    for d, sh, sm, eh, em in cases:
        s, e = rth_session_utc(d)
        assert s.hour == sh and s.minute == sm
        assert e.hour == eh and e.minute == em

