import time
import types

import pytest
from ai_trading import (
    alpaca_api,  # AI-AGENT-REF: canonical import
    utils,
)

try:
    from ai_trading.strategies.base import TradeSignal
except ImportError:  # pragma: no cover - optional strategies package
    pytest.skip("TradeSignal unavailable", allow_module_level=True)


def test_health_rows_throttle(monkeypatch, caplog):
    caplog.set_level("DEBUG")
    utils._last_health_log = 0.0
    t = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: t[0])
    utils.health_rows_passed([1])
    assert "HEALTH_ROWS_PASSED" in caplog.text
    caplog.clear()
    t[0] += 5
    utils.health_rows_passed([1])
    assert "HEALTH_ROWS_THROTTLED" in caplog.text


def test_cooldown_expired_throttle(monkeypatch, caplog):
    caplog.set_level("INFO")
    # Ensure we capture logs from the strategy_allocator module
    caplog.set_level("INFO", logger="strategy_allocator")
    import importlib

    from ai_trading import strategy_allocator  # AI-AGENT-REF: normalized import
    strategy_allocator = importlib.reload(strategy_allocator)
    alloc = strategy_allocator.StrategyAllocator()
    alloc.replace_config(signal_confirmation_bars=1)  # Allow single confirmation
    alloc.hold_protect = {"AAPL": 1}
    # Use a sell signal to test hold_protect functionality
    sig = TradeSignal(symbol="AAPL", side="sell", confidence=1.0, strategy="s")
    # Set last_direction to buy so we can test the direction change logic
    alloc.last_direction = {"AAPL": "buy"}
    t = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: t[0])

    alloc.allocate({"s": [sig]})
    assert any("HOLD_PROTECT_ACTIVE" in r.message for r in caplog.records)
    caplog.clear()
    alloc.hold_protect = {"AAPL": 1}
    alloc.last_direction = {"AAPL": "buy"}
    monkeypatch.setattr(time, "monotonic", lambda: t[0] + 5)
    alloc.allocate({"s": [sig]})
    assert any("HOLD_PROTECT_ACTIVE" in r.message for r in caplog.records)
    caplog.clear()
    alloc.hold_protect = {"AAPL": 0}  # Set to 0 so hold protect is not active
    alloc.last_direction = {"AAPL": "buy"}
    monkeypatch.setattr(time, "monotonic", lambda: t[0] + 20)
    alloc.allocate({"s": [sig]})
    assert not any("HOLD_PROTECT_ACTIVE" in r.message for r in caplog.records)


