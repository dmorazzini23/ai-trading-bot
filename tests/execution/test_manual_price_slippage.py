import os
import sys
import types

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
    monkeypatch.setattr("ai_trading.execution.engine.hash", lambda _: hash_value, raising=False)
    engine = ExecutionEngine()
    monkeypatch.setattr(engine, "_guess_price", lambda symbol: 100.0)
    monkeypatch.setattr(engine, "get_available_qty", lambda: 100)
    return engine


def test_manual_price_slippage_adjustment(monkeypatch, caplog):
    """Manual price orders should adjust without raising assertion errors."""

    engine = _make_engine(monkeypatch, 99)

    caplog.set_level("WARNING")

    order_id = engine.execute_order(
        "AAPL",
        OrderSide.BUY,
        10,
        order_type=OrderType.MARKET,
        price=100.0,
    )

    assert order_id is not None
    order = engine.order_manager.orders[order_id]

    assert order.order_type == OrderType.LIMIT
    assert order.quantity < 10

    warn_messages = [record.getMessage() for record in caplog.records]
    assert "SLIPPAGE_LIMIT_CONVERSION" in warn_messages
    assert "SLIPPAGE_QTY_REDUCED" in warn_messages


def test_predicted_slippage_improvement_buy_does_not_mitigate(monkeypatch, caplog):
    engine = _make_engine(monkeypatch, 0)

    caplog.set_level("WARNING")

    order_id = engine.execute_order(
        "AAPL",
        OrderSide.BUY,
        10,
        order_type=OrderType.MARKET,
        price=100.0,
    )

    assert order_id is not None
    order = engine.order_manager.orders[order_id]
    assert order.order_type == OrderType.MARKET
    assert order.quantity == 10
    warn_messages = [record.getMessage() for record in caplog.records]
    assert "SLIPPAGE_LIMIT_CONVERSION" not in warn_messages
    assert "SLIPPAGE_QTY_REDUCED" not in warn_messages


def test_predicted_slippage_improvement_sell_does_not_mitigate(monkeypatch, caplog):
    engine = _make_engine(monkeypatch, 99)

    caplog.set_level("WARNING")

    order_id = engine.execute_order(
        "AAPL",
        OrderSide.SELL,
        10,
        order_type=OrderType.MARKET,
        price=100.0,
    )

    assert order_id is not None
    order = engine.order_manager.orders[order_id]
    assert order.order_type == OrderType.MARKET
    assert order.quantity == 10
    warn_messages = [record.getMessage() for record in caplog.records]
    assert "SLIPPAGE_LIMIT_CONVERSION" not in warn_messages
    assert "SLIPPAGE_QTY_REDUCED" not in warn_messages


def test_predicted_slippage_adverse_sell_still_mitigates(monkeypatch, caplog):
    engine = _make_engine(monkeypatch, 0)

    caplog.set_level("WARNING")

    order_id = engine.execute_order(
        "AAPL",
        OrderSide.SELL,
        10,
        order_type=OrderType.MARKET,
        price=100.0,
    )

    assert order_id is not None
    order = engine.order_manager.orders[order_id]
    assert order.order_type == OrderType.LIMIT
    assert order.quantity < 10
    warn_messages = [record.getMessage() for record in caplog.records]
    assert "SLIPPAGE_LIMIT_CONVERSION" in warn_messages
    assert "SLIPPAGE_QTY_REDUCED" in warn_messages
