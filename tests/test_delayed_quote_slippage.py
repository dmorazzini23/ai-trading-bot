import pytest

import sys
from types import SimpleNamespace

from ai_trading.execution import ExecutionEngine
from ai_trading.core.enums import OrderSide


def test_delayed_quote_slippage_flagged(monkeypatch):
    """Large move between quote and fill should trigger slippage alert."""
    monkeypatch.setenv("TESTING", "false")
    monkeypatch.setenv("MAX_SLIPPAGE_BPS", "50")
    prices = iter([100.0, 102.0])
    stub = SimpleNamespace(get_latest_price=lambda symbol: next(prices))
    monkeypatch.setitem(sys.modules, "ai_trading.core.bot_engine", stub)
    monkeypatch.setattr("ai_trading.execution.engine.hash", lambda x: 50, raising=False)

    def fake_submit(self, order):
        self.orders[order.id] = order
        self.active_orders[order.id] = order
        return SimpleNamespace(id=order.id)

    monkeypatch.setattr(
        "ai_trading.execution.engine.OrderManager.submit_order",
        fake_submit,
        raising=False,
    )
    engine = ExecutionEngine()
    with pytest.raises(AssertionError):
        engine.execute_order("AAPL", OrderSide.BUY, 10)
    order = next(iter(engine.order_manager.orders.values()))
    assert round(order.slippage_bps, 2) > 50


def test_delayed_quote_slippage_within_threshold(monkeypatch):
    """Minor quote movement should record slippage without alert."""
    monkeypatch.setenv("TESTING", "false")
    monkeypatch.setenv("MAX_SLIPPAGE_BPS", "50")
    prices = iter([100.0, 100.3])
    stub = SimpleNamespace(get_latest_price=lambda symbol: next(prices))
    monkeypatch.setitem(sys.modules, "ai_trading.core.bot_engine", stub)
    monkeypatch.setattr("ai_trading.execution.engine.hash", lambda x: 50, raising=False)

    def fake_submit(self, order):
        self.orders[order.id] = order
        self.active_orders[order.id] = order
        return SimpleNamespace(id=order.id)

    monkeypatch.setattr(
        "ai_trading.execution.engine.OrderManager.submit_order",
        fake_submit,
        raising=False,
    )
    engine = ExecutionEngine()
    order_id = engine.execute_order("AAPL", OrderSide.BUY, 10)
    assert order_id is not None
    order = engine.order_manager.orders[order_id]
    assert round(float(order.expected_price), 2) == 100.0
    assert round(order.slippage_bps, 2) == 30.0
    assert abs(round(order.slippage_bps, 2)) < 50
