from __future__ import annotations

from datetime import UTC, datetime, timedelta

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

    monkeypatch.setattr(bars_mod, "http_get_bars", fake_http_get_bars)
    monkeypatch.setattr(bars_mod.time, "sleep", lambda *_: None)

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
