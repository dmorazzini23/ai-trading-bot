from __future__ import annotations

from datetime import UTC, date, datetime, time

from ai_trading.market import calendars
from ai_trading.market.calendars import AssetClass, CalendarRegistry, TradingSession


def test_equity_market_open_uses_session_timezone_for_utc_input() -> None:
    registry = CalendarRegistry()

    assert registry.is_market_open("AAPL", datetime(2025, 1, 2, 15, 0, tzinfo=UTC)) is True
    assert registry.is_market_open("AAPL", datetime(2025, 1, 2, 13, 0, tzinfo=UTC)) is False
    assert registry.is_market_open("AAPL", datetime(2025, 1, 1, 15, 0, tzinfo=UTC)) is False


def test_session_bounds_are_utc_for_regular_and_half_days() -> None:
    registry = CalendarRegistry()

    start, end = registry.get_session_bounds("SPY", date(2025, 7, 2))
    assert start == datetime(2025, 7, 2, 13, 30, tzinfo=UTC)
    assert end == datetime(2025, 7, 2, 20, 0, tzinfo=UTC)

    half_start, half_end = registry.get_session_bounds("SPY", date(2025, 7, 3))
    assert half_start == datetime(2025, 7, 3, 13, 30, tzinfo=UTC)
    assert half_end == datetime(2025, 7, 3, 17, 0, tzinfo=UTC)
    assert registry.is_market_open("SPY", datetime(2025, 7, 3, 17, 1, tzinfo=UTC)) is False


def test_custom_symbol_registration_and_asset_class_sessions() -> None:
    registry = CalendarRegistry()
    session = TradingSession(
        name="CUSTOM_UTC",
        start_time=time(10, 0),
        end_time=time(11, 0),
        days_of_week={0, 1, 2, 3, 4},
        timezone_name="UTC",
    )

    registry.register_symbol("test", session)

    assert registry.get_session("TEST") is session
    assert registry.is_market_open("test", datetime(2025, 1, 2, 10, 30, tzinfo=UTC)) is True
    assert registry.is_market_open("test", datetime(2025, 1, 2, 11, 30, tzinfo=UTC)) is False
    assert registry.get_session("BTCUSD").name == "CRYPTO_24_7"
    assert registry.get_session("SPY").name == "US_ETF_REGULAR"
    assert registry.get_session("UNKNOWN", AssetClass.BOND).name == "US_BOND_REGULAR"


def test_next_trading_day_and_final_bar_helpers(monkeypatch) -> None:
    registry = CalendarRegistry()
    registry.add_holiday(date(2025, 1, 6))

    assert registry.get_next_trading_day("AAPL", date(2025, 1, 3)) == date(2025, 1, 7)
    assert registry.ensure_final_bar("AAPL", "1min") is True
    assert registry.ensure_final_bar("AAPL", "1day") is False
    assert registry.ensure_final_bar("BTCUSD", "1min") is False

    monkeypatch.setattr(calendars, "_global_calendar", registry)
    assert calendars.is_trading_day("AAPL", datetime(2025, 1, 7, 15, 0, tzinfo=UTC)) is True
    assert calendars.ensure_final_bar("AAPL", "5min") is True


def test_generated_fallback_holidays_cover_2026_new_years_day() -> None:
    registry = CalendarRegistry()

    assert registry.is_trading_day("AAPL", date(2026, 1, 1)) is False
    assert registry.get_session_bounds("AAPL", date(2026, 1, 1)) == (None, None)


def test_crypto_remains_open_on_us_holidays_and_equity_holidays_are_lazy() -> None:
    registry = CalendarRegistry()

    assert registry.is_market_open("BTCUSD", datetime(2025, 1, 1, 15, 0, tzinfo=UTC)) is True
    assert registry.is_trading_day("AAPL", date(2031, 1, 1)) is False
    assert 2031 in registry._holiday_years  # noqa: SLF001
