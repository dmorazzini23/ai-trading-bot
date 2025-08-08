"""Critical test for type conversion bug in order fill reconciliation.

This test specifically validates the string-to-numeric conversion fix 
for the production TypeError in _reconcile_partial_fills.
"""

import pytest
import os
from unittest.mock import MagicMock, patch

# Set up test environment
os.environ.update({
    'ALPACA_API_KEY': 'test_key',
    'ALPACA_SECRET_KEY': 'test_secret', 
    'ALPACA_BASE_URL': 'https://paper-api.alpaca.markets',
    'WEBHOOK_SECRET': 'test_webhook',
    'FLASK_PORT': '5000'
})

from ai_trading.trade_execution import ExecutionEngine


class MockOrder:
    """Mock order object to simulate Alpaca order responses with various data types."""
    
    def __init__(self, filled_qty=None, status="filled", order_id="test-order-123"):
        self.filled_qty = filled_qty  # Can be string, int, float, or None
        self.status = status
        self.id = order_id
        self.symbol = "TEST"


class MockContext:
    """Mock trading context for testing."""
    
    def __init__(self):
        self.api = MagicMock()
        self.data_client = MagicMock()
        self.data_fetcher = MagicMock()
        self.capital_band = "small"


@pytest.mark.smoke
def test_string_filled_qty_conversion():
    """Test that string filled_qty values are properly converted to avoid TypeError."""
    
    ctx = MockContext()
    engine = ExecutionEngine(ctx)
    
    with patch.object(engine, 'logger') as mock_logger:
        
        # Test Case 1: String filled_qty that should work
        order_string_qty = MockOrder(filled_qty="75")  # String "75" 
        
        # This should NOT raise TypeError after the fix
        try:
            engine._reconcile_partial_fills(
                symbol="AAPL",
                requested_qty=100,
                remaining_qty=25,
                side="buy",
                last_order=order_string_qty
            )
            # If we get here, the fix worked
            success = True
        except TypeError as e:
            if "'>' not supported between instances of 'str' and 'int'" in str(e):
                success = False
                pytest.fail(f"TypeError still occurs with string input: {e}")
            else:
                raise  # Re-raise if it's a different TypeError
        except Exception as e:
            # Other exceptions are acceptable for this test
            success = True
        
        assert success, "String filled_qty conversion should not cause TypeError"


@pytest.mark.smoke
def test_various_filled_qty_types():
    """Test filled_qty with different data types: string, int, float, None."""
    
    ctx = MockContext()
    engine = ExecutionEngine(ctx)
    
    test_cases = [
        ("10", "string number"),
        (10, "integer"),
        (10.0, "float"),
        (None, "None value"),
        ("0", "string zero"),
        (0, "integer zero"),
        ("", "empty string"),
        ("invalid", "non-numeric string"),
    ]
    
    for filled_qty_value, description in test_cases:
        with patch.object(engine, 'logger') as mock_logger:
            order = MockOrder(filled_qty=filled_qty_value)
            
            # None of these should raise TypeError
            try:
                engine._reconcile_partial_fills(
                    symbol="TEST",
                    requested_qty=100,
                    remaining_qty=50,
                    side="buy",
                    last_order=order
                )
                print(f"✓ {description}: no TypeError")
            except TypeError as e:
                if "'>' not supported between instances of 'str' and 'int'" in str(e):
                    pytest.fail(f"TypeError with {description} ({filled_qty_value}): {e}")
                else:
                    raise  # Re-raise if it's a different TypeError
            except Exception:
                # Other exceptions are okay for this test
                print(f"✓ {description}: handled gracefully")


@pytest.mark.smoke
def test_string_qty_logging():
    """Test that type conversion failures are properly logged."""
    
    ctx = MockContext()
    engine = ExecutionEngine(ctx)
    
    with patch.object(engine, 'logger') as mock_logger:
        
        # Test with non-numeric string
        order_invalid = MockOrder(filled_qty="invalid_number")
        
        engine._reconcile_partial_fills(
            symbol="MSFT",
            requested_qty=50,
            remaining_qty=25,
            side="buy",
            last_order=order_invalid
        )
        
        # Check if the conversion failure was logged
        warning_calls = mock_logger.warning.call_args_list
        conversion_failed_logs = [
            call for call in warning_calls 
            if len(call[0]) > 0 and "ORDER_FILLED_QTY_CONVERSION_FAILED" in call[0][0]
        ]
        
        # Should log conversion failure for invalid strings
        assert len(conversion_failed_logs) > 0, "Should log conversion failure for invalid strings"


if __name__ == "__main__":
    # Run a quick test manually
    test_string_filled_qty_conversion()
    test_various_filled_qty_types()
    test_string_qty_logging()
    print("All critical type conversion tests completed")