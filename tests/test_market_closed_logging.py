"""Tests for market closed logging behaviors."""

import logging
import sys
import time
import types
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pd = None

from ai_trading.utils import base as utils


class DummyCalendar:
    def schedule(self, start_date, end_date):  # noqa: D401 - short helper
        """Return an empty schedule DataFrame for tests."""

        return pd.DataFrame()


def _install_dummy_calendar(monkeypatch):
    module = types.SimpleNamespace(get_calendar=lambda name: DummyCalendar())
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", module)


def test_logs_closed_once_per_date(monkeypatch, caplog):
    """is_market_open should log closed only once per day."""

    if pd is None:  # pragma: no cover - depends on optional pandas
        pytest.skip("pandas is required for calendar schedule stubs")
    _install_dummy_calendar(monkeypatch)
    monkeypatch.setattr(utils, "_LAST_MARKET_CLOSED_DATE", None)
    monkeypatch.setattr(utils, "_LAST_MARKET_HOURS_LOG", 0.0)
    monkeypatch.setattr(utils, "_LAST_MARKET_STATE", "")
    monkeypatch.setattr(
        utils,
        "_LAST_MARKET_STATE_DATES",
        {"OPEN": None, "CLOSED": None},
    )

    with caplog.at_level(logging.DEBUG):
        sat = datetime(2024, 1, 6, 10, tzinfo=ZoneInfo("America/New_York"))
        utils.is_market_open(sat)
        utils.is_market_open(sat)
        sun = datetime(2024, 1, 7, 10, tzinfo=ZoneInfo("America/New_York"))
        utils.is_market_open(sun)

    msgs = [r.getMessage() for r in caplog.records if "Detected Market Hours today: CLOSED" in r.getMessage()]
    assert len(msgs) == 2


def test_consecutive_closed_days_log_each_day(monkeypatch, caplog):
    """Ensure closed logs emit on consecutive closed days even without state change."""

    if pd is None:  # pragma: no cover - depends on optional pandas
        pytest.skip("pandas is required for calendar schedule stubs")

    _install_dummy_calendar(monkeypatch)
    monkeypatch.setattr(utils, "_LAST_MARKET_CLOSED_DATE", None)
    monkeypatch.setattr(utils, "_LAST_MARKET_STATE", "CLOSED")
    monkeypatch.setattr(utils, "_LAST_MARKET_HOURS_LOG", time.time())
    monkeypatch.setattr(
        utils,
        "_LAST_MARKET_STATE_DATES",
        {"OPEN": None, "CLOSED": datetime(2024, 1, 5, tzinfo=ZoneInfo("America/New_York")).date()},
    )

    with caplog.at_level(logging.DEBUG):
        sat = datetime(2024, 1, 6, 10, tzinfo=ZoneInfo("America/New_York"))
        utils.is_market_open(sat)
        utils._LAST_MARKET_HOURS_LOG = time.time()
        sun = datetime(2024, 1, 7, 10, tzinfo=ZoneInfo("America/New_York"))
        utils.is_market_open(sun)

    msgs = [r.getMessage() for r in caplog.records if "Detected Market Hours today: CLOSED" in r.getMessage()]
    assert len(msgs) == 2
    assert utils._LAST_MARKET_STATE_DATES["CLOSED"] == datetime(2024, 1, 7, tzinfo=ZoneInfo("America/New_York")).date()


def test_market_closed_sleep_is_capped(monkeypatch, caplog):
    from ai_trading import main as main_module

    caplog.set_level(logging.WARNING, logger=main_module.logger.name)

    stub_settings = types.SimpleNamespace(
        interval_when_closed=120,
        alpaca_data_feed="sip",
        alpaca_adjustment="raw",
    )

    monkeypatch.setattr(main_module, "ensure_dotenv_loaded", lambda: None)
    monkeypatch.setattr(main_module, "_check_alpaca_sdk", lambda: None)
    monkeypatch.setattr(main_module, "_fail_fast_env", lambda: None)
    monkeypatch.setattr(main_module, "_validate_runtime_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_module, "_init_http_session", lambda *args, **kwargs: False)
    monkeypatch.setattr(main_module, "get_settings", lambda: stub_settings)
    monkeypatch.setattr(main_module, "_is_market_open_base", lambda: False)

    sleep_calls = []

    def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(main_module.time, "sleep", fake_sleep)
    monkeypatch.setattr(main_module, "config", None, raising=False)

    delta = timedelta(hours=10)
    monkeypatch.setattr(main_module, "next_market_open", lambda now: now + delta)

    main_module.main([])

    assert sleep_calls == [pytest.approx(120.0)]

    records = [r for r in caplog.records if r.msg == "MARKET_CLOSED_SLEEP"]
    assert records, "expected MARKET_CLOSED_SLEEP warning"
    record = records[-1]
    assert record.sleep_original_s == int(delta.total_seconds())
    assert record.sleep_s == 120
    assert record.sleep_cap_s == 120
