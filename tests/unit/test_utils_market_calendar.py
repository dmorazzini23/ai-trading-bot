from __future__ import annotations

from datetime import date

import pytest

from ai_trading.utils import market_calendar


def test_fallback_calendar_skips_known_holidays(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(market_calendar, "_CAL", None)
    monkeypatch.setattr(market_calendar, "load_pandas_market_calendars", lambda: None)

    assert market_calendar.is_trading_day(date(2024, 1, 1)) is False
    assert market_calendar.is_trading_day(date(2026, 1, 1)) is False
    assert market_calendar.is_trading_day(date(2026, 1, 19)) is False
    assert market_calendar.is_trading_day(date(2026, 4, 3)) is False
    assert market_calendar.is_trading_day(date(2026, 7, 3)) is False


def test_fallback_previous_trading_session_skips_holidays(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(market_calendar, "_CAL", None)
    monkeypatch.setattr(market_calendar, "load_pandas_market_calendars", lambda: None)

    assert market_calendar.previous_trading_session(date(2024, 1, 2)) == date(2023, 12, 29)
    assert market_calendar.previous_trading_session(date(2026, 1, 20)) == date(2026, 1, 16)
