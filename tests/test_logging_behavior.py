import types
import time
import pytest

import utils
import bot_engine
from strategy_allocator import StrategyAllocator
import alpaca_api
from strategies.base import TradeSignal


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


def test_skip_cooldown_throttle(monkeypatch, caplog):
    caplog.set_level("INFO")
    bot_engine._LAST_SKIP_CD_TIME = 0.0
    bot_engine._LAST_SKIP_SYMBOLS = frozenset()
    t = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: t[0])
    bot_engine.log_skip_cooldown(["AAPL"])
    assert "SKIP_COOLDOWN" in caplog.text
    caplog.clear()
    bot_engine.log_skip_cooldown(["AAPL"])
    assert not caplog.records
    caplog.clear()
    t[0] += 16
    bot_engine.log_skip_cooldown(["AAPL"])
    assert caplog.records
    caplog.clear()
    bot_engine.log_skip_cooldown(["MSFT"])
    assert caplog.records


def test_cooldown_expired_throttle(monkeypatch, caplog):
    caplog.set_level("INFO")
    import importlib
    import strategy_allocator
    strategy_allocator = importlib.reload(strategy_allocator)
    alloc = strategy_allocator.StrategyAllocator()
    alloc.hold_protect = {"AAPL": 1}
    sig = TradeSignal(symbol="AAPL", side="buy", confidence=1.0, strategy="s")
    t = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: t[0])
    alloc.allocate({"s": [sig]}, volatility=1)
    assert any("COOLDOWN_EXPIRED" in r.message for r in caplog.records)
    caplog.clear()
    alloc.hold_protect = {"AAPL": 1}
    monkeypatch.setattr(time, "monotonic", lambda: t[0] + 5)
    alloc.allocate({"s": [sig]}, volatility=1)
    assert not any("COOLDOWN_EXPIRED" in r.message for r in caplog.records)
    caplog.clear()
    alloc.hold_protect = {"AAPL": 1}
    monkeypatch.setattr(time, "monotonic", lambda: t[0] + 20)
    alloc.allocate({"s": [sig]}, volatility=1)
    assert any("COOLDOWN_EXPIRED" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_order_filled_once(monkeypatch, caplog):
    caplog.set_level("DEBUG")
    alpaca_api.partial_fill_tracker.clear()
    alpaca_api.partial_fills.clear()
    t = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: t[0])
    event_partial = types.SimpleNamespace(
        order=types.SimpleNamespace(id="1", symbol="AAPL", filled_qty=1, filled_avg_price=1.0),
        event="partial_fill",
    )
    await alpaca_api.handle_trade_update(event_partial)
    assert any("ORDER_PARTIAL_FILL" in r.message for r in caplog.records)
    caplog.clear()
    await alpaca_api.handle_trade_update(event_partial)
    assert not any("ORDER_PARTIAL_FILL" in r.message for r in caplog.records)
    caplog.clear()
    event_fill = types.SimpleNamespace(
        order=types.SimpleNamespace(id="1", symbol="AAPL", filled_qty=2, filled_avg_price=1.2),
        event="fill",
    )
    await alpaca_api.handle_trade_update(event_fill)
    assert any("ORDER_FILLED" in r.message for r in caplog.records)

