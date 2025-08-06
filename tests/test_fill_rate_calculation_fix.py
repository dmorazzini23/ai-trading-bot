"""Test fill rate calculation fixes and alert thresholds."""

import pytest
import types
import logging
import os
from unittest.mock import MagicMock, patch

# Ensure test environment
os.environ.update({
    'ALPACA_API_KEY': 'test_key',
    'ALPACA_SECRET_KEY': 'test_secret', 
    'ALPACA_BASE_URL': 'https://paper-api.alpaca.markets',
    'WEBHOOK_SECRET': 'test_webhook',
    'FLASK_PORT': '5000'
})

from trade_execution import ExecutionEngine


class MockOrder:
    """Mock order object to simulate Alpaca order responses."""
    
    def __init__(self, filled_qty=None, status="filled", order_id="test-order-123"):
        self.filled_qty = filled_qty
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
def test_fill_rate_calculation_fix():
    """Test that fill rate calculation now works correctly when order.filled_qty is None."""
    
    ctx = MockContext()
    engine = ExecutionEngine(ctx)
    
    with patch.object(engine, 'logger') as mock_logger:
        
        # Test Case: Order with filled_qty=None (the bug condition that was fixed)
        order_without_filled_qty = MockOrder(filled_qty=None)
        
        engine._reconcile_partial_fills(
            symbol="QQQ",
            requested_qty=100,
            remaining_qty=50,  # 50 remaining = 50 filled out of 100
            side="buy", 
            last_order=order_without_filled_qty
        )
        
        logged_calls = mock_logger.warning.call_args_list
        partial_fill_logs = [call for call in logged_calls if call[0][0] == "PARTIAL_FILL_DETECTED"]
        
        assert len(partial_fill_logs) > 0, "Should have logged partial fill"
        
        log_extra = partial_fill_logs[0][1]['extra']
        
        # This should now be 50 filled out of 100 = 50%
        assert log_extra['filled_qty'] == 50, f"Expected filled_qty=50, got {log_extra['filled_qty']}"
        assert log_extra['fill_rate_pct'] == 50.0, f"Expected 50% fill rate, got {log_extra['fill_rate_pct']}"


@pytest.mark.smoke 
def test_fill_rate_alert_thresholds_updated():
    """Test that fill rate alert thresholds are now more realistic for market conditions."""
    
    ctx = MockContext()
    engine = ExecutionEngine(ctx)
    
    with patch.object(engine, 'logger') as mock_logger:
        
        # Test 50% fill rate - should NOT trigger any error alerts now
        order_50pct = MockOrder(filled_qty=25)
        
        engine._reconcile_partial_fills(
            symbol="SPY",
            requested_qty=50,
            remaining_qty=25,
            side="buy",
            last_order=order_50pct
        )
        
        error_calls = mock_logger.error.call_args_list
        low_fill_alerts = [call for call in error_calls if call[0][0] == "LOW_FILL_RATE_ALERT"]
        
        assert len(low_fill_alerts) == 0, "50% fill rate should not trigger LOW_FILL_RATE_ALERT"
        
        # Test 30% fill rate - should trigger moderate warning but not error
        mock_logger.reset_mock()
        order_30pct = MockOrder(filled_qty=15)
        
        engine._reconcile_partial_fills(
            symbol="AMZN",
            requested_qty=50, 
            remaining_qty=35,
            side="buy",
            last_order=order_30pct
        )
        
        warning_calls = mock_logger.warning.call_args_list
        moderate_alerts = [call for call in warning_calls if any("MODERATE_FILL_RATE_ALERT" in str(arg) for arg in call)]
        
        assert len(moderate_alerts) > 0, "30% fill rate should trigger MODERATE_FILL_RATE_ALERT"
        
        # Test 20% fill rate - should now trigger error-level alert
        mock_logger.reset_mock() 
        order_20pct = MockOrder(filled_qty=10)
        
        engine._reconcile_partial_fills(
            symbol="MSFT",
            requested_qty=50,
            remaining_qty=40,
            side="buy", 
            last_order=order_20pct
        )
        
        error_calls = mock_logger.error.call_args_list
        low_fill_alerts = [call for call in error_calls if call[0][0] == "LOW_FILL_RATE_ALERT"]
        
        assert len(low_fill_alerts) > 0, "20% fill rate should trigger LOW_FILL_RATE_ALERT"


@pytest.mark.smoke
def test_fill_rate_calculation_with_valid_order_data():
    """Test that fill rate calculation still works when order.filled_qty is properly set."""
    
    ctx = MockContext()
    engine = ExecutionEngine(ctx)
    
    with patch.object(engine, 'logger') as mock_logger:
        
        # Test Case: Order with valid filled_qty
        order_with_filled_qty = MockOrder(filled_qty=75)  # 75 out of 100 requested
        
        engine._reconcile_partial_fills(
            symbol="TSLA", 
            requested_qty=100,
            remaining_qty=25,  # This should be ignored since order has filled_qty
            side="buy",
            last_order=order_with_filled_qty
        )
        
        logged_calls = mock_logger.warning.call_args_list
        partial_fill_logs = [call for call in logged_calls if call[0][0] == "PARTIAL_FILL_DETECTED"]
        
        if partial_fill_logs:
            log_extra = partial_fill_logs[0][1]['extra']
            # Should use order.filled_qty (75) not calculated value (75)
            assert log_extra['filled_qty'] == 75, f"Expected filled_qty=75, got {log_extra['filled_qty']}"
            assert log_extra['fill_rate_pct'] == 75.0, f"Expected 75% fill rate, got {log_extra['fill_rate_pct']}"