import pytest
from types import SimpleNamespace

from ai_trading.execution.engine import (
    ExecutionEngine,
    ExecutionResult,
    Order,
    OrderManager,
)
from ai_trading.core.enums import OrderSide
from ai_trading.core.enums import OrderStatus
from ai_trading.risk.engine import TradeSignal


class DummyRiskEngine:
    def __init__(self):
        self.fills = []

    def register_fill(self, signal):
        self.fills.append(signal)


def _accepting_submit(order_manager, order):
    order_manager.orders[order.id] = order
    order_manager.active_orders[order.id] = order
    return SimpleNamespace(id=order.id)


def test_execute_order_returns_execution_result(monkeypatch):
    engine = ExecutionEngine()

    def submit_and_fill(self, order):
        self.orders[order.id] = order
        self.active_orders[order.id] = order
        order.add_fill(order.quantity, 101.0)
        return SimpleNamespace(id=order.id)

    monkeypatch.setattr(OrderManager, "submit_order", submit_and_fill, raising=False)
    monkeypatch.setattr(ExecutionEngine, "_simulate_market_execution", lambda self, order: None, raising=False)

    result = engine.execute_order("AAPL", OrderSide.BUY, 5)
    assert isinstance(result, ExecutionResult)
    assert isinstance(result, str)
    assert result.has_fill
    assert result.filled_quantity == 5
    assert result.fill_ratio == pytest.approx(1.0)


def test_async_fill_triggers_risk_engine(monkeypatch):
    risk = DummyRiskEngine()
    ctx = SimpleNamespace(risk_engine=risk)
    engine = ExecutionEngine(ctx=ctx)

    monkeypatch.setattr(OrderManager, "submit_order", _accepting_submit, raising=False)
    monkeypatch.setattr(ExecutionEngine, "_simulate_market_execution", lambda self, order: None, raising=False)

    signal = TradeSignal(
        symbol="AAPL",
        side="buy",
        confidence=1.0,
        strategy="s",
        weight=0.5,
        asset_class="equity",
    )

    result = engine.execute_order(
        "AAPL",
        OrderSide.BUY,
        10,
        signal=signal,
        signal_weight=signal.weight,
    )
    assert isinstance(result, ExecutionResult)
    assert result.filled_quantity == 0
    assert risk.fills == []

    order = engine.order_manager.orders[str(result)]
    order.add_fill(4, 100.0)
    engine._handle_execution_event(order, "completed")
    assert len(risk.fills) == 1
    assert pytest.approx(risk.fills[0].weight, rel=1e-6) == 0.2

    remaining = order.remaining_quantity
    order.add_fill(remaining, 100.0)
    engine._handle_execution_event(order, "completed")
    assert len(risk.fills) == 2
    assert pytest.approx(sum(sig.weight for sig in risk.fills), rel=1e-6) == pytest.approx(0.5)


def test_order_add_fill_tracks_partial_then_full():
    order = Order("AAPL", OrderSide.BUY, 5)

    order.add_fill(2, 101.0)
    assert order.status == OrderStatus.PARTIALLY_FILLED
    assert order.filled_quantity == 2

    order.add_fill(3, 102.0)
    assert order.status == OrderStatus.FILLED
    assert order.filled_quantity == 5


def test_simulate_market_execution_skips_canceled_unfilled_order():
    engine = ExecutionEngine()
    order = Order("AAPL", OrderSide.BUY, 5)
    order.status = OrderStatus.CANCELED

    filled_before = engine.execution_stats["filled_orders"]
    engine._simulate_market_execution(order)

    assert order.filled_quantity == 0
    assert engine.execution_stats["filled_orders"] == filled_before
