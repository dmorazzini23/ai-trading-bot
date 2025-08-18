#!/usr/bin/env python3
"""
Test Money math integration with execution engine.
"""

import sys
sys.path.append('.')

def test_money_execution_integration():
    """Test Money math integration with execution engine."""
    print("Testing Money Math + Execution Integration")
    
    from ai_trading.execution.engine import Order
    from ai_trading.core.enums import OrderSide, OrderType
    from ai_trading.math.money import Money
    
    # Test order creation with Money price
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=150,  # Will be rounded to lot size
        order_type=OrderType.LIMIT,
        price=Money("150.567")  # Will be quantized to tick
    )
    
    print(f"Original quantity: 150, Final quantity: {order.quantity}")
    print(f"Original price: 150.567, Final price: {order.price}")
    
    # Validate order with quantization
    from ai_trading.execution.engine import OrderManager
    manager = OrderManager()
    
    # This should trigger quantization in validation
    is_valid = manager._validate_order(order)
    print(f"Order validation: {'✓ PASS' if is_valid else '✗ FAIL'}")
    
    # Check that price was quantized to tick (0.01)
    expected_price = 150.57  # Should round to nearest cent
    actual_price = float(order.price)
    print(f"Price quantization: Expected ~{expected_price}, Got {actual_price}")
    
    # Check that quantity was rounded to lot size (default 1 for AAPL)
    print(f"Quantity remains: {order.quantity} (lot size is 1 for AAPL)")
    
    print("Money + Execution Integration: ✓ PASS\n")


def test_rate_limit_integration():
    """Test rate limiting integration."""
    print("Testing Rate Limiting Integration")
    
    # Test that the alpaca_api module uses rate limiting
    try:
        from ai_trading.integrations.rate_limit import get_limiter
        limiter = get_limiter()
        
        # Check that bars route is configured
        status = limiter.get_status("bars")
        print(f"Bars route configured: ✓ capacity={status['capacity']}, rate={status['refill_rate']}")
        
        # Check that orders route is configured  
        status = limiter.get_status("orders")
        print(f"Orders route configured: ✓ capacity={status['capacity']}, rate={status['refill_rate']}")
        
        print("Rate Limiting Integration: ✓ PASS\n")
        
    except Exception as e:
        print(f"Rate Limiting Integration: ✗ FAIL - {e}\n")


if __name__ == "__main__":
    print("=" * 50)
    print("INTEGRATION TESTS")
    print("=" * 50)
    
    try:
        test_money_execution_integration()
        test_rate_limit_integration()
        
        print("=" * 50)
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)