#!/usr/bin/env python3
"""
Focused test for short selling implementation.
Tests the specific changes needed to enable short selling capability.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

# Set minimal environment variables
os.environ['ALPACA_API_KEY'] = 'test_key'
os.environ['ALPACA_SECRET_KEY'] = 'test_secret'
os.environ['ALPACA_BASE_URL'] = 'https://paper-api.alpaca.markets'
os.environ['WEBHOOK_SECRET'] = 'test_webhook'
os.environ['FLASK_PORT'] = '9000'

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

class TestShortSellingImplementation(unittest.TestCase):
    """Test short selling capability implementation."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock dependencies to avoid import issues during testing
        self.mock_api = Mock()
        self.mock_account = Mock()
        self.mock_asset = Mock()
        
        # Configure mock responses
        self.mock_account.buying_power = "50000.0"
        self.mock_asset.shortable = True
        self.mock_asset.shortable_shares = 1000
        
        self.mock_api.get_account.return_value = self.mock_account
        self.mock_api.get_asset.return_value = self.mock_asset
        
    def test_current_sell_logic_blocks_no_position(self):
        """Test that current logic blocks sell orders when no position exists."""
        # This test documents the current behavior that we need to fix
        from trade_execution import ExecutionEngine
        
        # Create mock context
        mock_ctx = Mock()
        mock_ctx.api = self.mock_api
        
        engine = ExecutionEngine(mock_ctx)
        engine.logger = Mock()
        
        # Mock _available_qty to return 0 (no existing position)
        with patch.object(engine, '_available_qty', return_value=0):
            with patch.object(engine, '_select_api', return_value=self.mock_api):
                result = engine.execute_order("AAPL", 10, "sell")
                
                # Should return None due to SKIP_NO_POSITION logic
                self.assertIsNone(result)
                
                # Should log the skip message
                engine.logger.info.assert_called_with("SKIP_NO_POSITION | no shares to sell, skipping")
    
    def test_sell_short_validation_exists(self):
        """Test that _validate_short_selling method exists and works."""
        from trade_execution import ExecutionEngine
        
        # Create mock context
        mock_ctx = Mock()
        mock_ctx.api = self.mock_api
        
        engine = ExecutionEngine(mock_ctx)
        engine.logger = Mock()
        
        # Test that validation method exists
        self.assertTrue(hasattr(engine, '_validate_short_selling'))
        
        # Test validation logic
        result = engine._validate_short_selling(self.mock_api, "AAPL", 10)
        self.assertTrue(result)  # Should pass with our mock setup
        
    def test_sell_short_side_should_be_distinguished(self):
        """Test that sell_short orders bypass position checks and validate short selling."""
        from trade_execution import ExecutionEngine
        
        # Create mock context
        mock_ctx = Mock()
        mock_ctx.api = self.mock_api
        
        engine = ExecutionEngine(mock_ctx)
        engine.logger = Mock()
        
        # Test the key distinction: sell vs sell_short in early validation
        # Mock everything to prevent execution from proceeding too far
        with patch.object(engine, '_available_qty', return_value=0):
            with patch.object(engine, '_select_api', return_value=self.mock_api):
                with patch.object(engine, '_validate_short_selling', return_value=True):
                    with patch.object(engine, '_assess_liquidity', side_effect=Exception("Stop execution here")):
                        
                        # Test that sell_short orders reach the validation step (don't get blocked by SKIP_NO_POSITION)
                        try:
                            result = engine.execute_order("AAPL", 10, "sell_short")
                        except Exception:
                            # Expected to reach this point, meaning it passed the initial validation
                            pass
                        
                        # Verify short selling validation was called
                        engine._validate_short_selling.assert_called_once_with(self.mock_api, "AAPL", 10)
                        # Verify the short sell initiation log
                        engine.logger.info.assert_any_call("SHORT_SELL_INITIATED | symbol=%s qty=%d", "AAPL", 10)
                        
                        # Now test that regular sell orders are blocked when no position exists
                        engine._validate_short_selling.reset_mock()
                        engine.logger.reset_mock()
                        
                        result = engine.execute_order("AAPL", 10, "sell")
                        
                        # Should return None and log SKIP_NO_POSITION
                        self.assertIsNone(result)
                        engine.logger.info.assert_called_with("SKIP_NO_POSITION | no shares to sell, skipping")
                        # Short selling validation should NOT be called for regular sell
                        engine._validate_short_selling.assert_not_called()
        
    def test_order_status_monitoring_needed(self):
        """Test framework for order status monitoring."""
        from trade_execution import ExecutionEngine
        
        # Create mock context
        mock_ctx = Mock()
        mock_ctx.api = self.mock_api
        
        engine = ExecutionEngine(mock_ctx)
        engine.logger = Mock()
        
        # Test order tracking functionality
        mock_order = Mock()
        mock_order.id = "test_order_123"
        mock_order.status = "new"
        
        # Test _track_order method
        engine._track_order(mock_order, "AAPL", "buy", 10)
        
        # Verify order is tracked
        pending_orders = engine.get_pending_orders()
        self.assertEqual(len(pending_orders), 1)
        self.assertEqual(pending_orders[0].order_id, "test_order_123")
        self.assertEqual(pending_orders[0].symbol, "AAPL")
        self.assertEqual(pending_orders[0].side, "buy")
        
        # Test status update
        engine._update_order_status("test_order_123", "filled")
        
        # Verify order is removed from tracking after terminal status
        pending_orders = engine.get_pending_orders()
        self.assertEqual(len(pending_orders), 0)
        
        # Test stale order cleanup
        # Add an old order
        old_order = Mock()
        old_order.id = "old_order_456"
        old_order.status = "new"
        engine._track_order(old_order, "MSFT", "sell", 5)
        
        # Mock the order as old by manipulating the tracking directly
        import time
        from trade_execution import _active_orders, _order_tracking_lock
        with _order_tracking_lock:
            if "old_order_456" in _active_orders:
                _active_orders["old_order_456"].submitted_time = time.time() - 700  # 700 seconds ago
        
        # Mock the cancel method to avoid API calls
        with patch.object(engine, '_cancel_stale_order', return_value=True):
            canceled_count = engine.cleanup_stale_orders(max_age_seconds=600)  # 10 minutes
            self.assertEqual(canceled_count, 1)

    def test_meta_learning_graceful_degradation(self):
        """Test that meta-learning provides graceful degradation when no data exists."""
        from ai_trading.core.bot_engine import load_global_signal_performance
        
        # Test when no trade log file exists
        with patch('os.path.exists', return_value=False):
            result = load_global_signal_performance()
            # Should return None gracefully instead of raising an error
            self.assertIsNone(result)
        
        # Test when trade log exists but is empty or has insufficient data
        with patch('os.path.exists', return_value=True):
            with patch('pandas.read_csv') as mock_read_csv:
                # Mock empty dataframe
                import pandas as pd
                mock_read_csv.return_value = pd.DataFrame()
                
                result = load_global_signal_performance(min_trades=1)  # Lower threshold
                # Should return empty dict instead of None for empty data
                self.assertEqual(result, {})

if __name__ == '__main__':
    unittest.main()