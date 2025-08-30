from ai_trading.execution.engine import OrderManager, Order
from ai_trading.core.enums import OrderSide, OrderType


def test_submit_order_returns_object_with_filled_qty():
    om = OrderManager()
    order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET)
    result = om.submit_order(order)
    assert result is not None, "submit_order should return an object"
    assert result.id == order.id
    assert result.symbol == "AAPL"
    assert result.side == OrderSide.BUY.value
    assert str(getattr(result, "filled_qty")) == "0"

