from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from ai_trading.position.trailing_stops import TrailingStopManager


def _make_position(entry: float, qty: int, trail_pct: float | None = None) -> SimpleNamespace:
    payload: dict[str, float | int] = {
        "avg_entry_price": entry,
        "qty": qty,
    }
    if trail_pct is not None:
        payload["trail_pct"] = trail_pct
    return SimpleNamespace(**payload)


def test_trailing_stop_initialization_and_ratchet(caplog: pytest.LogCaptureFixture) -> None:
    manager = TrailingStopManager()
    position = _make_position(100.0, 10)
    caplog.set_level(logging.INFO)
    level = manager.update_trailing_stop("AAPL", position, 101.0)
    assert level is not None
    assert level.stop_price < 101.0
    assert level.side == "long"
    assert not level.is_triggered

    # Ratchet upward as price makes new highs.
    level = manager.update_trailing_stop("AAPL", position, 110.0)
    first_stop = level.stop_price
    level = manager.update_trailing_stop("AAPL", position, 115.0)
    assert level.stop_price >= first_stop


def test_trailing_stop_correction_logged(caplog: pytest.LogCaptureFixture) -> None:
    manager = TrailingStopManager()
    position = _make_position(50.0, 5)
    level = manager.update_trailing_stop("MSFT", position, 55.0)
    # Introduce inconsistent state to force correction.
    level.stop_price = 500.0
    caplog.set_level(logging.INFO)
    manager.update_trailing_stop("MSFT", position, 56.0)
    corrections = [rec for rec in caplog.records if rec.msg == "TRAILING_STOP_CORRECTED"]
    assert corrections, "expected correction log"
    payload = corrections[-1].__dict__
    assert payload["side"] == "long"
    assert pytest.approx(level.trail_pct) == payload["trail_pct"]
    assert "max_since_entry" in payload


def test_trailing_stop_trigger_logs_details(caplog: pytest.LogCaptureFixture) -> None:
    manager = TrailingStopManager()
    position = _make_position(30.0, 8)
    level = manager.update_trailing_stop("TSLA", position, 36.0)
    target_stop = float(level.stop_price)
    caplog.clear()
    caplog.set_level(logging.WARNING)
    # Price crosses the stop level which should trigger the stop.
    manager.update_trailing_stop("TSLA", position, target_stop * 0.99)
    triggered = [rec for rec in caplog.records if rec.msg == "TRAILING_STOP_TRIGGERED"]
    assert triggered, "expected trailing stop trigger log"
    payload = triggered[-1].__dict__
    assert payload["side"] == "long"
    assert payload["reason"] == "price_crossed_stop"
    assert "max_since_entry" in payload
