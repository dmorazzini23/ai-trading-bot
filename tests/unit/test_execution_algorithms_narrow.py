from ai_trading.execution.algorithms import TWAPExecutor
from ai_trading.execution.engine import OrderManager
from ai_trading.core.enums import OrderSide, OrderType


class DummyOrderManager(OrderManager):
    # Use OrderManager behavior; submit_order returns False/True safely.
    pass


def test_twap_zero_duration_returns_empty_list():
    om = DummyOrderManager()
    twap = TWAPExecutor(om)
    orders = twap.execute_twap_order(
        symbol="TEST",
        side=OrderSide.BUY,
        total_quantity=1000,
        duration_minutes=0,  # triggers zero-division path before loop
    )
    assert orders == []

