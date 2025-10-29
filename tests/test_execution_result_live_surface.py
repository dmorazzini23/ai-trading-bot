"""Tests for :class:`ai_trading.execution.engine.ExecutionResult`."""

from ai_trading.execution.engine import ExecutionResult, Order
from ai_trading.core.enums import OrderSide, OrderType


def test_execution_result_exposes_side_and_symbol_for_buy():
    order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=5, order_type=OrderType.MARKET)
    result = ExecutionResult(order, "accepted", 0, 5, None)

    assert result.side == "buy"
    assert result.symbol == "AAPL"


def test_execution_result_side_maps_short_variants_to_sell():
    order = Order(symbol="TSLA", side=OrderSide.SELL_SHORT, quantity=1, order_type=OrderType.MARKET)
    result = ExecutionResult(order, "accepted", 0, 1, None)

    assert result.side == "sell"
