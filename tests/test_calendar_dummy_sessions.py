from datetime import date

import ai_trading.market.calendar_wrapper as cw


def test_dummy_sessions_cover_2024_jan(monkeypatch):
    """Fallback sessions for early 2024 should provide RTH windows."""
    monkeypatch.setattr(cw, "load_pandas_market_calendars", lambda: None)
    monkeypatch.setattr(cw, "_CAL", None)

    for d in [date(2024, 1, 1), date(2024, 1, 2)]:
        start, end = cw.rth_session_utc(d)
        assert start.hour == 14 and start.minute == 30
        assert end.hour == 21 and end.minute == 0
