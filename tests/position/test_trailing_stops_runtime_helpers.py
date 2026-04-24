from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from ai_trading.position.trailing_stops import (
    TrailingStopLevel,
    TrailingStopManager,
    TrailingStopType,
    _fmt,
    _to_float,
)


def _market_frame(rows: int = 40, *, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    close = [start + i * step for i in range(rows)]
    return pd.DataFrame(
        {
            "close": close,
            "high": [price + 2.0 for price in close],
            "low": [price - 2.0 for price in close],
            "volume": [1000 + i for i in range(rows)],
        }
    )


def _position(entry: float = 100.0, qty: int = 10, trail_pct: object | None = None) -> SimpleNamespace:
    payload: dict[str, object] = {"avg_entry_price": entry, "qty": qty}
    if trail_pct is not None:
        payload["trail_pct"] = trail_pct
    return SimpleNamespace(**payload)


def _level(
    *,
    side: str = "long",
    current_price: float = 110.0,
    stop_price: float = 105.0,
    entry_price: float = 100.0,
    trail_pct: float = 0.03,
) -> TrailingStopLevel:
    return TrailingStopLevel(
        symbol="AAPL",
        current_price=current_price,
        stop_price=stop_price,
        stop_type=TrailingStopType.ADAPTIVE,
        trail_distance=3.0,
        max_price_achieved=max(entry_price, current_price),
        entry_price=entry_price,
        unrealized_gain_pct=(current_price - entry_price) / entry_price * 100,
        days_held=0,
        last_updated=datetime.now(UTC),
        side=side,
        trail_pct=trail_pct,
        max_price_since_entry=max(entry_price, current_price),
        min_price_since_entry=min(entry_price, current_price),
    )


def test_format_and_float_helpers_handle_bad_values() -> None:
    assert _fmt(1.2345678) == "1.234568"
    assert _fmt(cast(Any, "bad")) == "nan"
    assert _to_float("3.5") == 3.5
    assert _to_float("bad") is None
    assert _to_float(float("nan")) is None


def test_update_trailing_stop_handles_invalid_positions_and_short_side() -> None:
    manager = TrailingStopManager()

    assert manager.update_trailing_stop("AAPL", None, 100.0) is None
    assert manager.update_trailing_stop("AAPL", _position(entry=0.0), 100.0) is None
    assert manager.update_trailing_stop("AAPL", _position(qty=0), 100.0) is None

    short_level = manager.update_trailing_stop("TSLA", _position(entry=100.0, qty=-5), 95.0)

    assert short_level is not None
    assert short_level.side == "short"
    assert short_level.stop_price > 95.0


def test_update_trailing_stop_resets_state_when_side_flips() -> None:
    manager = TrailingStopManager()
    manager.update_trailing_stop("AAPL", _position(entry=100.0, qty=10), 110.0)

    flipped = manager.update_trailing_stop("AAPL", _position(entry=100.0, qty=-10), 95.0)

    assert flipped is not None
    assert flipped.side == "short"
    assert flipped.min_price_since_entry == 95.0
    assert flipped.stop_price > flipped.current_price


def test_stop_level_accessors_and_remove() -> None:
    manager = TrailingStopManager()
    level = manager.update_trailing_stop("AAPL", _position(), 110.0)
    assert manager.get_stop_level("AAPL") is level

    level.is_triggered = True
    assert manager.get_triggered_stops() == [level]

    manager.remove_stop_level("AAPL")
    assert manager.get_stop_level("AAPL") is None


def test_initialize_stop_level_uses_default_for_bad_trail_pct() -> None:
    manager = TrailingStopManager()

    level = manager._initialize_stop_level("AAPL", 100.0, 110.0, _position(trail_pct="bad"))  # noqa: SLF001

    assert level.trail_pct == pytest.approx(0.03)
    assert level.stop_price < level.current_price


def test_adaptive_stop_uses_market_data_and_short_fallback(monkeypatch) -> None:
    manager = TrailingStopManager()
    level = _level(current_price=110.0, entry_price=100.0)
    monkeypatch.setattr(manager, "_get_market_data", lambda _symbol: _market_frame())

    long_stop = manager._calculate_adaptive_stop("AAPL", level)  # noqa: SLF001

    assert long_stop < level.current_price

    short_like = _level(side="short", current_price=90.0, entry_price=100.0)
    monkeypatch.setattr(manager, "_get_market_data", lambda _symbol: None)

    short_stop = manager._calculate_adaptive_stop("AAPL", short_like)  # noqa: SLF001

    assert short_stop > short_like.current_price


def test_merge_and_directional_correction_for_long_and_short() -> None:
    manager = TrailingStopManager()

    assert manager._merge_stop_prices(100.0, 105.0, "long") == 105.0  # noqa: SLF001
    assert manager._merge_stop_prices(100.0, 95.0, "short") == 95.0  # noqa: SLF001
    assert manager._merge_stop_prices(cast(Any, "bad"), 95.0, "short") == 95.0  # noqa: SLF001

    long_level = _level(side="long", current_price=100.0, stop_price=105.0)
    short_level = _level(side="short", current_price=100.0, stop_price=95.0)

    assert manager._ensure_directional_stop(long_level, 100.0, candidate=105.0) is True  # noqa: SLF001
    assert long_level.stop_price < 100.0
    assert manager._ensure_directional_stop(short_level, 100.0, candidate=95.0) is True  # noqa: SLF001
    assert short_level.stop_price > 100.0


def test_initial_atr_momentum_time_and_breakeven_helpers(monkeypatch) -> None:
    manager = TrailingStopManager()
    data = _market_frame(rows=30, start=100.0, step=0.5)
    monkeypatch.setattr(manager, "_get_market_data", lambda _symbol: data)

    assert manager._calculate_initial_stop_distance("AAPL") >= manager.base_trail_percent  # noqa: SLF001
    assert manager._calculate_atr_stop_distance(data) >= 1.0  # noqa: SLF001
    assert manager._calculate_atr_stop_distance(pd.DataFrame({"close": [1.0]})) == manager.base_trail_percent  # noqa: SLF001

    monkeypatch.setattr(manager, "_calculate_rsi", lambda *_args: 80.0)
    assert manager._calculate_momentum_multiplier("AAPL", data) == 1.3  # noqa: SLF001
    monkeypatch.setattr(manager, "_calculate_rsi", lambda *_args: 20.0)
    assert manager._calculate_momentum_multiplier("AAPL", data) == 0.7  # noqa: SLF001
    monkeypatch.setattr(manager, "_calculate_rsi", lambda *_args: 50.0)
    assert manager._calculate_momentum_multiplier("AAPL", data) == 1.0  # noqa: SLF001
    assert manager._calculate_momentum_multiplier("AAPL", None) == 1.0  # noqa: SLF001

    assert manager._calculate_time_decay_multiplier(7) == 1.0  # noqa: SLF001
    assert manager._calculate_time_decay_multiplier(22) == pytest.approx(0.75)  # noqa: SLF001
    assert manager._calculate_time_decay_multiplier(cast(Any, "bad")) == 1.0  # noqa: SLF001

    assert manager._calculate_breakeven_distance(_level(current_price=110.0, entry_price=100.0)) is not None  # noqa: SLF001
    assert manager._calculate_breakeven_distance(_level(current_price=101.0, entry_price=100.0)) is None  # noqa: SLF001


def test_combine_stop_distances_weights_and_bounds() -> None:
    manager = TrailingStopManager()
    level = _level()

    combined = manager._combine_stop_distances(  # noqa: SLF001
        {"fixed": 3.0, "atr": 5.0, "momentum": 4.0, "time_decay": 2.0},
        level,
    )
    breakeven = manager._combine_stop_distances(  # noqa: SLF001
        {"fixed": 3.0, "atr": 5.0, "momentum": 4.0, "time_decay": 2.0, "breakeven": 0.1},
        level,
    )
    fallback = manager._combine_stop_distances({"unknown": 50.0}, level)  # noqa: SLF001
    bounded = manager._combine_stop_distances({"fixed": 100.0}, level)  # noqa: SLF001

    assert combined > 0
    assert breakeven < combined
    assert fallback == manager.base_trail_percent
    assert bounded == 15.0


def test_check_stop_trigger_for_short_and_prior_reference() -> None:
    manager = TrailingStopManager()
    short_level = _level(side="short", current_price=105.0, stop_price=104.0)

    assert manager._check_stop_trigger(short_level, -10) is True  # noqa: SLF001
    assert short_level.is_triggered is True
    assert short_level.trigger_reason == "price_crossed_stop"

    long_level = _level(side="long", current_price=102.0, stop_price=95.0)
    assert manager._check_stop_trigger(long_level, 10, prior_stop_price=103.0) is True  # noqa: SLF001

    invalid = _level(stop_price=100.0)
    invalid.stop_price = "bad"  # type: ignore[assignment]
    assert manager._check_stop_trigger(invalid, 10) is False  # noqa: SLF001


def test_market_data_and_rsi_helpers() -> None:
    minute = _market_frame(rows=20)
    daily = _market_frame(rows=10)

    class _Fetcher:
        def get_minute_df(self, _ctx, _symbol):
            return minute

        def get_daily_df(self, _ctx, _symbol):
            return daily

    manager = TrailingStopManager(SimpleNamespace(data_fetcher=_Fetcher()))

    assert manager._get_market_data("AAPL") is minute  # noqa: SLF001

    prices = pd.Series([float(i) for i in range(1, 40)])
    assert 0 <= manager._calculate_rsi(prices, 14) <= 100  # noqa: SLF001
    assert manager._calculate_rsi([1.0, 2.0], 14) == 50.0  # noqa: SLF001
    assert manager._calculate_rsi(object(), 14) == 50.0  # noqa: SLF001
    assert manager._calculate_days_held(SimpleNamespace()) == 0  # noqa: SLF001
