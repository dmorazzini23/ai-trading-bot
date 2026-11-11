"""is_market_open must honour NYSE early closes from the exchange calendar."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from ai_trading.utils import base as utils

pytest.importorskip("pandas_market_calendars")

NY = ZoneInfo("America/New_York")


@pytest.fixture(autouse=True)
def _reset_market_state(monkeypatch):
    monkeypatch.setattr(utils, "_LAST_MARKET_CLOSED_DATE", None)
    monkeypatch.setattr(utils, "_LAST_MARKET_HOURS_LOG", 0.0)
    monkeypatch.setattr(utils, "_LAST_MARKET_STATE", "")
    monkeypatch.setattr(utils, "_LAST_MARKET_STATE_DATES", {"OPEN": None, "CLOSED": None})
    monkeypatch.setenv("FORCE_MARKET_OPEN", "false")


@pytest.mark.parametrize("day", [(2025, 7, 3), (2028, 7, 3), (2026, 11, 27), (2026, 12, 24)])
def test_early_close_days_are_closed_after_one_pm(day):
    assert utils.is_market_open(datetime(*day, 12, 30, tzinfo=NY)) is True
    assert utils.is_market_open(datetime(*day, 14, 30, tzinfo=NY)) is False


def test_regular_day_stays_open_until_four_pm():
    assert utils.is_market_open(datetime(2026, 7, 2, 15, 30, tzinfo=NY)) is True
