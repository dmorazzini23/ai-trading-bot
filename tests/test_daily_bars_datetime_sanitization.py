from __future__ import annotations

from datetime import UTC, datetime, timedelta, date

import types

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.data import bars as bars_mod
from ai_trading.data import fetch as fetch_mod
from ai_trading.data.bars import StockBarsRequest, TimeFrame, safe_get_stock_bars


def test_request_timestamps_sanitized_and_passed_to_get_bars():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 2, tzinfo=UTC)
    req = StockBarsRequest(
        symbol_or_symbols="SPY",
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed="sip",
    )

    captured: dict[str, str] = {}

    class DummyClient:
        def get_bars(self, symbol_or_symbols, timeframe, **params):
            # Only record the first call containing start/end; subsequent
            # fallback calls omit these parameters.
            if "start" in params or "end" in params:
                captured.update({"start": params.get("start"), "end": params.get("end")})
            return pd.DataFrame()

    safe_get_stock_bars(DummyClient(), req, symbol="SPY", context="TEST")

    expected_end = end.replace(hour=23, minute=59, second=59, microsecond=999999)
    assert req.start == start
    assert req.end == expected_end
    assert captured["start"] == start.isoformat()
    assert captured["end"] == expected_end.isoformat()


def test_request_timestamps_sanitized_for_get_stock_bars():
    start = datetime(2024, 1, 3, tzinfo=UTC)
    end = datetime(2024, 1, 4, tzinfo=UTC)
    req = StockBarsRequest(
        symbol_or_symbols="SPY",
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed="sip",
    )

    captured: dict[str, str] = {}

    class DummyClient:
        class Resp:
            df = pd.DataFrame()

        def get_stock_bars(self, request):  # pragma: no cover - simple stub
            if getattr(request, "start", None) is not None:
                captured["start"] = request.start
                captured["end"] = request.end
            return self.Resp()

    safe_get_stock_bars(DummyClient(), req, symbol="SPY", context="TEST")

    expected_end = end.replace(hour=23, minute=59, second=59, microsecond=999999)
    assert captured["start"] == start
    assert captured["end"] == expected_end


def test_safe_get_stock_bars_sets_datetimes_on_plain_request():
    start = datetime(2024, 1, 7, 13, tzinfo=UTC)
    end = datetime(2024, 1, 8, 12, tzinfo=UTC)

    class PlainRequest:
        symbol_or_symbols = "SPY"
        timeframe = "1Day"
        feed = "sip"

        def __init__(self):
            self.start = start
            self.end = end

    captured: dict[str, datetime] = {}

    class DummyClient:
        def get_stock_bars(self, request):
            captured["start"] = request.start
            captured["end"] = request.end
            return pd.DataFrame({"open": [1], "high": [1], "low": [1], "close": [1]})

    req = PlainRequest()

    safe_get_stock_bars(DummyClient(), req, symbol="SPY", context="TEST")

    expected_start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    expected_end = end.replace(hour=23, minute=59, second=59, microsecond=999999)

    assert isinstance(req.start, datetime)
    assert req.start == expected_start
    assert req.start.tzinfo == UTC
    assert isinstance(req.end, datetime)
    assert req.end == expected_end
    assert req.end.tzinfo == UTC
    assert captured["start"] == req.start
    assert captured["end"] == expected_end


def test_http_fallback_receives_iso_timestamps(monkeypatch):
    monkeypatch.setenv("ENABLE_HTTP_FALLBACK", "force")
    start = datetime(2024, 1, 9, tzinfo=UTC)
    end = datetime(2024, 1, 10, tzinfo=UTC)
    req = StockBarsRequest(
        symbol_or_symbols="SPY",
        timeframe=TimeFrame.Hour,
        start=start,
        end=end,
        feed="sip",
    )

    captured_http: dict[str, str] = {}

    def fake_http_get_bars(symbol, timeframe, start, end, *, feed=None):
        captured_http["start"] = start
        captured_http["end"] = end
        return pd.DataFrame()

    class DummyClient:
        class Resp:
            df = pd.DataFrame()

        def get_stock_bars(self, request):
            return self.Resp()

    def fail_fetch(*_args, **_kwargs):  # pragma: no cover - exercised indirectly
        raise RuntimeError("fail")

    monkeypatch.setattr(bars_mod, "http_get_bars", fake_http_get_bars)
    monkeypatch.setattr(bars_mod.time, "sleep", lambda *_: None)
    monkeypatch.setattr(bars_mod, "_client_fetch_stock_bars", fail_fetch)

    safe_get_stock_bars(DummyClient(), req, symbol="SPY", context="TEST")

    assert isinstance(req.start, datetime)
    assert isinstance(req.end, datetime)
    assert captured_http["start"] == req.start.isoformat()
    assert captured_http["end"] == req.end.isoformat()


def test_window_has_trading_session_handles_missing_holiday_session(monkeypatch):
    freeze_time = pytest.importorskip("freezegun").freeze_time

    holiday = datetime(2024, 7, 4, 12, tzinfo=UTC)
    start = holiday - timedelta(hours=1)
    end = holiday + timedelta(hours=1)

    with freeze_time("2024-07-04 12:00:00", tz_offset=0):
        holiday_date = holiday.date()

        def fake_is_trading_day(day):
            return day == holiday_date

        def fake_rth_session_utc(day):
            raise RuntimeError("calendar missing session")

        monkeypatch.setattr(fetch_mod, "is_trading_day", fake_is_trading_day)
        monkeypatch.setattr(fetch_mod, "rth_session_utc", fake_rth_session_utc)

        assert fetch_mod._window_has_trading_session(start, end) is False


def test_session_info_uses_fallback_when_calendar_available(monkeypatch):
    mc = pytest.importorskip("ai_trading.utils.market_calendar")

    holiday = date(2024, 1, 1)
    previous = date(2023, 12, 29)

    class DummyCalendar:
        def schedule(self, start_date, end_date):
            if start_date == previous and end_date == previous:
                return pd.DataFrame(
                    {
                        "market_open": [pd.Timestamp("2023-12-29 14:30", tz="UTC")],
                        "market_close": [pd.Timestamp("2023-12-29 21:00", tz="UTC")],
                    }
                )
            return pd.DataFrame(columns=["market_open", "market_close"])

    pmc = types.SimpleNamespace(get_calendar=lambda _: DummyCalendar())

    monkeypatch.setattr(mc, "_CAL", None)
    monkeypatch.setattr(mc, "load_pandas_market_calendars", lambda: pmc)
    monkeypatch.setattr(mc, "load_pandas", lambda: pd)

    session = mc.session_info(holiday)

    assert session.start_utc == datetime(2024, 1, 1, 14, 30, tzinfo=UTC)
    assert session.end_utc == datetime(2024, 1, 1, 21, 0, tzinfo=UTC)
    assert session.is_early_close is False


def test_is_trading_day_falls_back_without_valid_days(monkeypatch):
    mc = pytest.importorskip("ai_trading.utils.market_calendar")

    class NoValidDaysCalendar:
        def schedule(self, start_date, end_date):  # pragma: no cover - unused here
            return pd.DataFrame(columns=["market_open", "market_close"])

    pmc = types.SimpleNamespace(get_calendar=lambda _: NoValidDaysCalendar())

    monkeypatch.setattr(mc, "_CAL", None)
    monkeypatch.setattr(mc, "load_pandas_market_calendars", lambda: pmc)

    weekday = date(2024, 1, 2)
    weekend = date(2024, 1, 6)

    assert mc.is_trading_day(weekday) is True
    assert mc.is_trading_day(weekend) is False
