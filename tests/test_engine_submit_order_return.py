from ai_trading.execution.engine import OrderManager, Order
from ai_trading.core.enums import OrderSide, OrderType
from typing import Any, cast


def test_submit_order_returns_object_with_filled_qty():
    om = OrderManager()
    order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET)
    result = om.submit_order(order)
    assert result is not None, "submit_order should return an object"
    result_obj = cast(Any, result)
    assert result_obj.id == order.id
    assert result_obj.symbol == "AAPL"
    assert result_obj.side == OrderSide.BUY.value
    assert str(getattr(result_obj, "filled_qty")) == "0"
