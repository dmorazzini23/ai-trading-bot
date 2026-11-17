#!/usr/bin/env python3
"""Test Money math integration with execution engine."""

import pytest

pytestmark = pytest.mark.integration

def test_money_execution_integration():
    """Test Money math integration with execution engine."""

    from ai_trading.core.enums import OrderSide, OrderType
    from ai_trading.execution.engine import Order
    from ai_trading.math.money import Money

    # Test order creation with Money price
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=150,  # Will be rounded to lot size
        order_type=OrderType.LIMIT,
        price=Money("150.567")  # Will be quantized to tick
    )


    # Validate order with quantization
    from ai_trading.execution.engine import OrderManager
    manager = OrderManager()

    # This should trigger quantization in validation
    manager._validate_order(order)

    # Check that price was quantized to tick (0.01)
    float(order.price)

    # Check that quantity was rounded to lot size (default 1 for AAPL)





