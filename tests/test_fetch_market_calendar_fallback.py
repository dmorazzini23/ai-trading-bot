import logging

from ai_trading.data import fetch
from ai_trading.utils import base


def test_calendar_failure_assumes_open_and_is_logged_once(monkeypatch, caplog):
    def _broken(*_args, **_kwargs):
        raise RuntimeError("calendar offline")

    monkeypatch.setattr(base, "is_market_open", _broken)
    monkeypatch.setattr(fetch, "_MARKET_CALENDAR_FALLBACK_LOGGED", False)

    with caplog.at_level(logging.WARNING):
        assert fetch.is_market_open() is True
        assert fetch.is_market_open() is True

    fallbacks = [
        r for r in caplog.records
        if r.getMessage() == "DATA_MARKET_CALENDAR_UNAVAILABLE_ASSUMING_OPEN"
    ]
    assert len(fallbacks) == 1
    assert getattr(fallbacks[0], "cause", None) == "RuntimeError"
