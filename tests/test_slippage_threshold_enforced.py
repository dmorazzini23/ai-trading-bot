import os
import pytest

from ai_trading.execution import ExecutionEngine
from ai_trading.core.enums import OrderSide, OrderType
from ai_trading.core.constants import EXECUTION_PARAMETERS


def test_slippage_threshold_enforced(monkeypatch, caplog):
    os.environ["TESTING"] = "true"
    monkeypatch.setitem(EXECUTION_PARAMETERS, "MAX_SLIPPAGE_BPS", 10)
    engine = ExecutionEngine()
    monkeypatch.setattr("ai_trading.execution.engine.hash", lambda x: 99, raising=False)

    with caplog.at_level("WARNING"):
        order_id = engine.execute_order("AAPL", OrderSide.BUY, 10, price=100.0)

    assert order_id is not None
    order = engine.order_manager.orders[str(order_id)]
    assert order.order_type == OrderType.LIMIT
    assert order.quantity < 10
    warnings = [record.getMessage() for record in caplog.records]
    assert "SLIPPAGE_LIMIT_CONVERSION" in warnings
    assert "SLIPPAGE_QTY_REDUCED" in warnings
