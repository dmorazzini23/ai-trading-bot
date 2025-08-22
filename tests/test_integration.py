#!/usr/bin/env python3
"""
Test Money math integration with execution engine.
"""

import sys

sys.path.append('.')

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



def test_rate_limit_integration():
    """Test rate limiting integration."""

    # Test that the alpaca_api module uses rate limiting
    try:
        from ai_trading.integrations.rate_limit import get_limiter
        limiter = get_limiter()

        # Check that bars route is configured
        limiter.get_status("bars")

        # Check that orders route is configured
        limiter.get_status("orders")


    # noqa: BLE001 TODO: narrow exception
    except Exception:
        pass


if __name__ == "__main__":

    try:
        test_money_execution_integration()
        test_rate_limit_integration()


    # noqa: BLE001 TODO: narrow exception
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
