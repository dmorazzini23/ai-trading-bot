import datetime as dt
import sys
from zoneinfo import ZoneInfo

import pandas as pd

from ai_trading.utils import base as base_utils


class _FakeCalendar:
    def schedule(self, start_date, end_date):  # noqa: D401 - simple stub
        _ = (start_date, end_date)
        return pd.DataFrame({"unexpected": []})


class _FakeMCal:
    @staticmethod
    def get_calendar(name: str):
        assert name == "NYSE"
        return _FakeCalendar()


def test_next_market_open_missing_market_open_column(monkeypatch):
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", _FakeMCal())

    saturday = dt.datetime(2024, 1, 6, 12, tzinfo=dt.UTC)

    nxt = base_utils.next_market_open(saturday)

    assert nxt == dt.datetime(2024, 1, 8, 9, 30, tzinfo=ZoneInfo("America/New_York"))
