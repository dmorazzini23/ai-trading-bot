"""Tests for market closed logging once per date."""
from __future__ import annotations

import logging
import sys
import types
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from ai_trading.utils import base as utils


class DummyCalendar:
    def schedule(self, start_date, end_date):
        return pd.DataFrame()


def _install_dummy_calendar(monkeypatch):
    module = types.SimpleNamespace(get_calendar=lambda name: DummyCalendar())
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", module)


def test_logs_closed_once_per_date(monkeypatch, caplog):
    """is_market_open should log closed only once per day."""
    _install_dummy_calendar(monkeypatch)
    monkeypatch.setattr(utils, "_LAST_MARKET_CLOSED_DATE", None)
    monkeypatch.setattr(utils, "_LAST_MARKET_HOURS_LOG", 0.0)
    monkeypatch.setattr(utils, "_LAST_MARKET_STATE", "")

    with caplog.at_level(logging.DEBUG):
        sat = datetime(2024, 1, 6, 10, tzinfo=ZoneInfo("America/New_York"))
        utils.is_market_open(sat)
        utils.is_market_open(sat)
        sun = datetime(2024, 1, 7, 10, tzinfo=ZoneInfo("America/New_York"))
        utils.is_market_open(sun)

    msgs = [r.getMessage() for r in caplog.records if "Detected Market Hours today: CLOSED" in r.getMessage()]
    assert len(msgs) == 2
