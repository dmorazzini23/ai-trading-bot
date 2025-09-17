import os
import sys
import types

import pytest

cachetools_stub = types.ModuleType("cachetools")


class _TTLCache(dict):
    def __init__(self, maxsize, ttl):
        super().__init__()
        self.maxsize = maxsize
        self.ttl = ttl


cachetools_stub.TTLCache = _TTLCache
sys.modules.setdefault("cachetools", cachetools_stub)

from ai_trading.core.enums import OrderSide, OrderType
from ai_trading.execution import ExecutionEngine


def _make_engine(monkeypatch, hash_value):
    os.environ["TESTING"] = "true"
    monkeypatch.setenv("MAX_SLIPPAGE_BPS", "10")
    monkeypatch.setenv("SLIPPAGE_LIMIT_TOLERANCE_BPS", "5")
    monkeypatch.setattr(
        "ai_trading.execution.engine.hash", lambda _: hash_value, raising=False
    )
    monkeypatch.setattr(
        "ai_trading.execution.engine.time.sleep", lambda *_, **__: None, raising=False
    )
    monkeypatch.setattr(
        "ai_trading.execution.engine.OrderManager.start_monitoring",
        lambda self: None,
        raising=False,
    )
    engine = ExecutionEngine()
    engine.available_qty = 10_000
    monkeypatch.setattr(engine, "_guess_price", lambda symbol: 100.0, raising=False)
    monkeypatch.setattr(engine, "_update_position", lambda *_, **__: None, raising=False)
    return engine


def test_manual_price_slippage_adjustment(monkeypatch, caplog):
    """Manual price orders should adjust without raising assertion errors."""

    os.environ["TESTING"] = "true"
    monkeypatch.setenv("MAX_SLIPPAGE_BPS", "10")
    monkeypatch.setenv("SLIPPAGE_LIMIT_TOLERANCE_BPS", "5")
    monkeypatch.setattr("ai_trading.execution.engine.hash", lambda _: 99, raising=False)

    engine = ExecutionEngine()
    monkeypatch.setattr(engine, "_guess_price", lambda symbol: 100.0)

    caplog.set_level("WARNING")

    order_id = engine.execute_order(
        "AAPL",
        OrderSide.BUY,
        10,
        order_type=OrderType.MARKET,
        price=100.0,
    )

    order = engine.order_manager.orders[order_id]

    assert order.order_type == OrderType.LIMIT
    assert order.quantity < 10

    warn_messages = [record.getMessage() for record in caplog.records]
    assert "SLIPPAGE_LIMIT_CONVERSION" in warn_messages
    assert "SLIPPAGE_QTY_REDUCED" in warn_messages


def test_buy_slippage_improvement_allows_execution(monkeypatch, caplog):
    """A buy order with better-than-expected price should proceed."""

    engine = _make_engine(monkeypatch, hash_value=0)

    with caplog.at_level("WARNING"):
        result = engine.execute_order(
            "AAPL",
            OrderSide.BUY,
            10,
            order_type=OrderType.MARKET,
            price=100.0,
        )

    order = engine.order_manager.orders[str(result)]
    assert order.order_type == OrderType.MARKET
    assert order.quantity == 10
    warn_messages = [record.getMessage() for record in caplog.records]
    assert "SLIPPAGE_LIMIT_CONVERSION" not in warn_messages
    assert "SLIPPAGE_QTY_REDUCED" not in warn_messages
    assert "SLIPPAGE_ORDER_REJECTED" not in warn_messages


def test_sell_slippage_improvement_allows_execution(monkeypatch, caplog):
    """A sell order with positive slippage should not be mitigated."""

    engine = _make_engine(monkeypatch, hash_value=99)

    with caplog.at_level("WARNING"):
        result = engine.execute_order(
            "AAPL",
            OrderSide.SELL,
            10,
            order_type=OrderType.MARKET,
            price=100.0,
        )

    order = engine.order_manager.orders[str(result)]
    assert order.order_type == OrderType.MARKET
    assert order.quantity == 10
    warn_messages = [record.getMessage() for record in caplog.records]
    assert "SLIPPAGE_LIMIT_CONVERSION" not in warn_messages
    assert "SLIPPAGE_QTY_REDUCED" not in warn_messages
    assert "SLIPPAGE_ORDER_REJECTED" not in warn_messages


@pytest.mark.parametrize("hash_value, side", [(99, OrderSide.BUY), (0, OrderSide.SELL)])
def test_adverse_slippage_still_triggers_controls(monkeypatch, caplog, hash_value, side):
    """Directional slippage against the order should still be mitigated."""

    engine = _make_engine(monkeypatch, hash_value=hash_value)

    with caplog.at_level("WARNING"):
        result = engine.execute_order(
            "AAPL",
            side,
            10,
            order_type=OrderType.MARKET,
            price=100.0,
        )

    order = engine.order_manager.orders[str(result)]
    assert order.order_type == OrderType.LIMIT
    assert order.quantity < 10
    warn_messages = [record.getMessage() for record in caplog.records]
    assert "SLIPPAGE_LIMIT_CONVERSION" in warn_messages
    assert "SLIPPAGE_QTY_REDUCED" in warn_messages
