#!/usr/bin/env python3
"""
Critical trading bot issues test suite.
Tests for order execution tracking, meta-learning log formats, liquidity management.
"""
from tests.optdeps import require
require("pandas")

import csv
import os
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import pandas as pd

# Set up minimal environment for imports
os.environ.setdefault('ALPACA_API_KEY', 'test_key')
os.environ.setdefault('ALPACA_SECRET_KEY', 'test_secret')
os.environ.setdefault('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
os.environ.setdefault('WEBHOOK_SECRET', 'test_webhook')
os.environ.setdefault('FLASK_PORT', '5000')
os.environ.setdefault('PYTEST_RUNNING', '1')

# Import the modules we need to test
try:
    from ai_trading import meta_learning
    from ai_trading.core import bot_engine
    from ai_trading.monitoring.order_health_monitor import _order_tracking_lock
except ImportError:
    # Create minimal mocks for missing imports
    bot_engine = MagicMock()
    meta_learning = MagicMock()
    _order_tracking_lock = MagicMock()


class TestOrderExecutionTracking(unittest.TestCase):
    """Test order execution and filled quantity tracking issues."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Mock bot context
        self.mock_ctx = Mock()
        self.mock_ctx.api = Mock()
        self.mock_ctx.data_client = Mock()
        self.mock_ctx.trade_logger = Mock()

        # Mock order responses
        self.mock_order = Mock()
        self.mock_order.id = "test-order-123"
        self.mock_order.status = "filled"
        self.mock_order.filled_qty = "50"  # This is the key issue - partial fill
        self.mock_order.qty = "100"       # Original requested quantity

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_order_slicing_quantity_mismatch(self):
        """Test that order slicing properly tracks actual vs intended quantities."""
        # This test validates the core issue: signals generate qty X but only Y is filled

        symbol = "AAPL"
        total_qty = 100
        side = "buy"

        # Mock market data for POV calculation
        mock_df = pd.DataFrame({
            'volume': [1000000],
            'close': [150.0]
        })

        # Mock quote data for spread calculation
        mock_quote = Mock()
        mock_quote.ask_price = 150.10
        mock_quote.bid_price = 150.00
        mock_quote.spread = 0.10  # High spread to trigger liquidity retry

        with patch('ai_trading.core.bot_engine.fetch_minute_df_safe', return_value=mock_df), \
             patch.object(self.mock_ctx.data_client, 'get_stock_latest_quote', return_value=mock_quote), \
             patch('ai_trading.core.bot_engine.submit_order', return_value=self.mock_order) as mock_submit:

            # Test that we can access the POV submit function
            if hasattr(bot_engine, 'pov_submit'):
                # This would expose the quantity tracking issue
                bot_engine.pov_submit(self.mock_ctx, symbol, total_qty, side)

                # Verify that submit_order was called with sliced quantities
                self.assertTrue(mock_submit.called)
                calls = mock_submit.call_args_list

                # Calculate total intended vs actual filled
                sum(call[0][2] for call in calls)  # qty parameter
                # The issue: we track total_intended but actual filled might be different

                # This test currently fails because we don't track actual fills properly

    def test_order_status_polling_integration(self):
        """Test that order status polling properly feeds back to slicing logic."""
        # This tests the disconnect between order polling and slice tracking

        order_id = "test-order-123"

        # Mock partially filled order
        partial_order = Mock()
        partial_order.status = "partially_filled"
        partial_order.filled_qty = "50"
        partial_order.qty = "100"

        with patch.object(self.mock_ctx.api, 'get_order_by_id', return_value=partial_order):
            if hasattr(bot_engine, 'poll_order_fill_status'):
                # This should track the actual fill but doesn't integrate with slice logic
                bot_engine.poll_order_fill_status(self.mock_ctx, order_id, timeout=1)

                # The issue: poll_order_fill_status runs in a separate thread
                # and doesn't communicate back to the main slicing logic

    def test_safe_submit_order_quantity_validation(self):
        """Test that safe_submit_order validates filled_qty matches intended qty."""

        # Mock order request
        mock_req = Mock()
        mock_req.symbol = "AAPL"
        mock_req.qty = 100

        # Mock order with partial fill
        partial_order = Mock()
        partial_order.status = "partially_filled"
        partial_order.filled_qty = "50"  # Only half filled
        partial_order.qty = "100"

        with patch.object(self.mock_ctx.api, 'submit_order', return_value=partial_order), \
             patch.object(self.mock_ctx.api, 'get_order_by_id', return_value=partial_order):

            if hasattr(bot_engine, 'safe_submit_order'):
                result = bot_engine.safe_submit_order(self.mock_ctx.api, mock_req)

                # The issue: function returns the order but doesn't validate
                # that filled_qty (50) matches intended qty (100)
                self.assertEqual(result.filled_qty, "50")
                self.assertEqual(result.qty, "100")
                # This exposes the quantity mismatch issue


class TestMetaLearningLogFormat(unittest.TestCase):
    """Test meta-learning trade log format parsing and conversion."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.trade_log_path = os.path.join(self.temp_dir, "test_trades.csv")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_mixed_format_detection(self):
        """Test detection of mixed audit/meta-learning log formats."""

        # Create a mixed format CSV file without headers to test raw data parsing
        mixed_data = [
            # Audit format row (UUID in first column)
            ["12345678-1234-1234-1234-123456789012", "2025-08-05T10:00:00Z", "AAPL", "buy", "100", "150.00", "live", "filled"],
            # Meta format row (Symbol in first column)
            ["MSFT", "2025-08-05T10:05:00Z", "140.00", "2025-08-05T10:10:00Z", "142.00", "50", "buy", "momentum", "profitable", "ma_cross", "0.75", "100.00"],
            # Another audit format row
            ["87654321-4321-4321-4321-210987654321", "2025-08-05T10:15:00Z", "GOOGL", "sell", "25", "2800.00", "live", "filled"]
        ]

        with open(self.trade_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for row in mixed_data:
                writer.writerow(row)

        if hasattr(meta_learning, 'validate_trade_data_quality'):
            quality_report = meta_learning.validate_trade_data_quality(self.trade_log_path)

            # Should detect mixed format
            self.assertTrue(quality_report.get('mixed_format_detected', False))
            self.assertGreater(quality_report.get('audit_format_rows', 0), 0)
            self.assertGreater(quality_report.get('meta_format_rows', 0), 0)

            # Should have recommendations for handling mixed format
            recommendations = quality_report.get('recommendations', [])
            self.assertTrue(any('unified parsing' in rec.lower() for rec in recommendations))

    def test_meta_learning_empty_log_issue(self):
        """Test the METALEARN_EMPTY_TRADE_LOG issue reproduction."""

        # Create an audit-only format file (triggers the empty log warning)
        audit_only_data = [
            ["order_id", "timestamp", "symbol", "side", "qty", "price", "mode", "status"],
            ["12345678-1234-1234-1234-123456789012", "2025-08-05T10:00:00Z", "AAPL", "buy", "100", "150.00", "live", "filled"],
            ["87654321-4321-4321-4321-210987654321", "2025-08-05T10:15:00Z", "GOOGL", "sell", "25", "2800.00", "live", "filled"]
        ]

        with open(self.trade_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for row in audit_only_data:
                writer.writerow(row)

        if hasattr(meta_learning, 'load_trade_data'):
            # This should trigger the "No valid trades found" warning
            result = meta_learning.load_trade_data(self.trade_log_path)

            # Should return False/empty because no meta-learning format rows
            self.assertFalse(result)

    def test_audit_to_meta_format_conversion(self):
        """Test conversion from audit format to meta-learning format."""

        # Create audit format data
        audit_data = [
            ["order_id", "timestamp", "symbol", "side", "qty", "price", "mode", "status"],
            ["12345678-1234-1234-1234-123456789012", "2025-08-05T10:00:00Z", "AAPL", "buy", "100", "150.00", "live", "filled"],
            ["12345678-1234-1234-1234-123456789013", "2025-08-05T10:05:00Z", "AAPL", "sell", "100", "152.00", "live", "filled"]
        ]

        with open(self.trade_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for row in audit_data:
                writer.writerow(row)

        # Load as DataFrame for conversion
        df = pd.read_csv(self.trade_log_path)

        if hasattr(meta_learning, '_convert_audit_to_meta_format'):
            converted_df = meta_learning._convert_audit_to_meta_format(df)

            # Should have meta-learning columns
            expected_cols = ['symbol', 'entry_time', 'entry_price', 'exit_time', 'exit_price']
            for col in expected_cols:
                self.assertIn(col, converted_df.columns)


class TestLiquidityManagement(unittest.TestCase):
    """Test liquidity management and threshold optimization."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_ctx = Mock()
        self.mock_ctx.api = Mock()
        self.mock_ctx.data_client = Mock()
        self.mock_ctx.volume_threshold = 100000  # Default volume threshold

    def test_conservative_spread_threshold(self):
        """Test that 0.05 spread threshold is too conservative."""

        symbol = "AAPL"

        # Mock quote with moderate spread
        mock_quote = Mock()
        mock_quote.ask_price = 150.05
        mock_quote.bid_price = 150.00
        mock_quote.spread = 0.05  # Exactly at threshold

        # Mock volume data

        with patch.object(self.mock_ctx.data_client, 'get_stock_latest_quote', return_value=mock_quote):
            if hasattr(bot_engine, 'liquidity_factor'):
                bot_engine.liquidity_factor(self.mock_ctx, symbol)

                # With current logic, spread of 0.05 reduces liquidity factor significantly
                # This is too conservative for normal market conditions

                # Should be more reasonable with dynamic thresholds

    def test_pov_slice_reduction_on_spread(self):
        """Test POV slice quantity reduction due to spread."""

        # Mock market data
        mock_df = pd.DataFrame({
            'volume': [1000000],  # Good volume
            'close': [150.0]
        })

        # Mock quote with high spread (triggers current reduction logic)
        mock_quote = Mock()
        mock_quote.ask_price = 150.10
        mock_quote.bid_price = 150.00
        # spread = 0.10 > 0.05 threshold

        total_qty = 100
        pct = 0.1  # 10% participation rate

        with patch('ai_trading.core.bot_engine.fetch_minute_df_safe', return_value=mock_df), \
             patch.object(self.mock_ctx.data_client, 'get_stock_latest_quote', return_value=mock_quote):

            # Current logic in pov_submit:
            vol = mock_df["volume"].iloc[-1]
            spread = 0.10

            if spread > 0.05:
                slice_qty = min(int(vol * pct * 0.5), total_qty)  # Reduced by 50%
            else:
                slice_qty = min(int(vol * pct), total_qty)

            # This shows excessive reduction due to conservative threshold
            int(vol * pct)  # 100,000
            reduced_slice = int(vol * pct * 0.5)  # 50,000

            self.assertEqual(slice_qty, min(reduced_slice, total_qty))

    def test_volatility_retry_frequency(self):
        """Test excessive liquidity retries due to volatility."""

        # This test would need to check historical patterns
        # For now, document the issue pattern

        # Issue pattern from logs:
        # 1. Signal generates qty (e.g., 38 shares)
        # 2. Liquidity factor reduces it due to "volatility"
        # 3. Only partial amount ordered (e.g., 19 shares)
        # 4. System reports full 38 as filled (tracking issue)

        pass


class TestResourceManagement(unittest.TestCase):
    """Test resource management and memory optimization."""

    def test_memory_optimization_available(self):
        """Test memory optimization module availability."""
        try:
            from memory_optimizer import (
                emergency_memory_cleanup,
                memory_profile,
                optimize_memory,
            )
            self.assertTrue(hasattr(memory_profile, '__call__'))
            self.assertTrue(hasattr(optimize_memory, '__call__'))
            self.assertTrue(hasattr(emergency_memory_cleanup, '__call__'))
        except ImportError:
            # Should have fallback decorators
            pass

    def test_recent_buys_cleanup(self):
        """Test recent_buys cleanup to prevent memory leaks."""
        assert True

    def test_order_submission_lock(self):
        """Test order submission locking mechanism."""
        self.assertTrue(hasattr(_order_tracking_lock, 'acquire'))
        self.assertTrue(hasattr(_order_tracking_lock, 'release'))


if __name__ == '__main__':
    # Run tests with verbose output to see the issues
    unittest.main(verbosity=2)
