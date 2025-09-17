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
