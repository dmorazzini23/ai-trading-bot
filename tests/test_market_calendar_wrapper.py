from __future__ import annotations

# Tests for NYSE market calendar helpers.  # AI-AGENT-REF
from datetime import date

import pytest
from tests.optdeps import require
import ai_trading.utils.market_calendar as cw
from ai_trading.utils.market_calendar import is_trading_day, rth_session_utc

pmc = require("pandas_market_calendars")




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


def test_fallback_calendar_handles_2026_holidays(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cw, "_CAL", None)
    monkeypatch.setattr(cw, "load_pandas_market_calendars", lambda: None)

    assert is_trading_day(date(2026, 1, 1)) is False
    assert is_trading_day(date(2026, 1, 19)) is False
    assert is_trading_day(date(2026, 4, 3)) is False
    assert is_trading_day(date(2026, 7, 3)) is False
    assert cw.previous_trading_session(date(2026, 1, 19)) == date(2026, 1, 16)


def test_fallback_calendar_computes_future_early_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cw, "_CAL", None)
    monkeypatch.setattr(cw, "load_pandas_market_calendars", lambda: None)

    assert cw.is_early_close(date(2026, 11, 27)) is True
    assert cw.rth_session_utc(date(2026, 11, 27))[1].hour == 18
    assert cw.is_early_close(date(2026, 12, 24)) is True
