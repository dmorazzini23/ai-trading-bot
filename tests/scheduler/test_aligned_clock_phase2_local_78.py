from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.scheduler import aligned_clock as ac


class _Calendar:
    tz = ZoneInfo("America/New_York")

    def __init__(self, *, empty: bool = False, fail_schedule: bool = False) -> None:
        self.empty = empty
        self.fail_schedule = fail_schedule

    def valid_days(self, start_date, end_date):
        return pd.DatetimeIndex([pd.Timestamp(start_date) + pd.Timedelta(days=1)])

    def schedule(self, start_date, end_date):
        if self.fail_schedule:
            raise ValueError("bad schedule")
        if self.empty:
            return pd.DataFrame()
        day = pd.Timestamp(start_date)
        return pd.DataFrame(
            {
                "market_open": [pd.Timestamp(day.date()).replace(hour=14, minute=30, tzinfo=UTC)],
                "market_close": [pd.Timestamp(day.date()).replace(hour=21, minute=0, tzinfo=UTC)],
            }
        )


def test_calendar_loading_and_timeframe_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ac, "mcal", None)
    monkeypatch.setattr(ac.importlib, "import_module", lambda _name: SimpleNamespace(get_calendar=lambda name: _Calendar()))
    assert ac._get_calendar("NYSE") is not None  # noqa: SLF001

    class FixedClock(ac.AlignedClock):
        def __init__(self) -> None:
            super().__init__(max_skew_ms=1.0)
            self.calendar = None

        def get_exchange_time(self, tz=None):
            current = datetime(2026, 4, 27, 15, 59, 45, tzinfo=UTC)
            return current if tz is None else current.astimezone(tz)

    clock = FixedClock()
    assert clock._parse_timeframe_minutes("5m") == 5  # noqa: SLF001
    assert clock._parse_timeframe_minutes("1h") == 60  # noqa: SLF001
    assert clock._parse_timeframe_minutes("1d") == 1440  # noqa: SLF001
    with pytest.raises(ValueError):
        clock._parse_timeframe_minutes("bad")  # noqa: SLF001

    close_1m = clock.next_bar_close("AAPL", "1m")
    close_daily = clock.next_bar_close("AAPL", "1d")
    assert close_1m == datetime(2026, 4, 27, 16, 0, tzinfo=UTC)
    assert close_daily == datetime(2026, 4, 27, 16, 0, tzinfo=UTC)
    assert clock.next_bar_close("AAPL", "1m") is close_1m


def test_final_bar_market_open_wait_and_global_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = ac.AlignedClock(max_skew_ms=100.0)
    clock.calendar = _Calendar()
    current = datetime(2026, 4, 27, 15, 59, 59, 950000, tzinfo=UTC)
    monkeypatch.setattr(clock, "get_exchange_time", lambda tz=None: current)
    monkeypatch.setattr(clock, "next_bar_close", lambda symbol, timeframe="1m", tz=None: current + timedelta(milliseconds=50))
    monkeypatch.setattr(clock, "check_skew", lambda reference_time=None, tz=None: 0.0)

    validation = clock.ensure_final_bar("AAPL", "1m", tzinfo=UTC)
    assert validation.is_final is False
    assert validation.reason and "Too close" in validation.reason

    clock.calendar = None
    assert clock.is_market_open("AAPL", datetime(2026, 4, 27, 10, 0, tzinfo=UTC)) is True
    assert clock.is_market_open("AAPL", datetime(2026, 4, 25, 10, 0, tzinfo=UTC)) is False
    clock.calendar = _Calendar(empty=True)
    assert clock.is_market_open("AAPL", datetime(2026, 4, 27, 15, 0, tzinfo=UTC)) is False
    clock.calendar = _Calendar(fail_schedule=True)
    assert clock.is_market_open("AAPL", datetime(2026, 4, 27, 15, 0, tzinfo=UTC)) is False

    attempts = iter(
        [
            ac.BarValidation("AAPL", "1m", False, current, current, 0.0),
            ac.BarValidation("AAPL", "1m", True, current, current, 0.0),
        ]
    )
    monkeypatch.setattr(clock, "ensure_final_bar", lambda *args, **kwargs: next(attempts))
    monkeypatch.setattr(ac.time, "time", lambda: 0.0)
    monkeypatch.setattr(ac.time, "sleep", lambda _seconds: None)
    assert clock.wait_for_aligned_tick("AAPL").is_final is True

    monkeypatch.setattr(
        clock,
        "ensure_final_bar",
        lambda *args, **kwargs: ac.BarValidation("AAPL", "1m", True, current, current, 0.0),
    )
    monkeypatch.setattr(ac, "_global_clock", clock)
    assert ac.get_aligned_clock() is clock
    assert ac.ensure_final_bar("AAPL").is_final is True
    assert ac.is_market_open("AAPL", datetime(2026, 4, 27, 10, 0, tzinfo=UTC)) is False
