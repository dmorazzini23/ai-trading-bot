from datetime import date

import pytest

import ai_trading.utils.market_calendar as cw


def test_dummy_sessions_cover_2024_jan(monkeypatch):
    """Fallback sessions for early 2024 should provide RTH windows."""
    monkeypatch.setattr(cw, "load_pandas_market_calendars", lambda: None)
    monkeypatch.setattr(cw, "_CAL", None)

    with pytest.raises(ValueError, match="not_trading_session"):
        cw.rth_session_utc(date(2024, 1, 1))

    start, end = cw.rth_session_utc(date(2024, 1, 2))
    assert start.hour == 14 and start.minute == 30
    assert end.hour == 21 and end.minute == 0
