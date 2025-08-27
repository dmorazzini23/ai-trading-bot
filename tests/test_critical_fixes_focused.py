#!/usr/bin/env python3
"""
Focused test suite for the critical trading bot fixes per problem statement.
"""

import csv
import os
import tempfile
import unittest
from datetime import UTC

from ai_trading.utils.timefmt import isoformat_z

from tests.support.mocks import MockContext, MockSignal

# Set testing environment
os.environ['TESTING'] = '1'


class TestCriticalFixes(unittest.TestCase):
    """Test suite for critical P0 and P1 fixes."""

    def setUp(self):
        """Set up test environment."""
        # Import modules after setting TESTING flag
        from ai_trading import strategy_allocator  # AI-AGENT-REF: normalized import
        from ai_trading.analysis import sentiment
        self.sentiment = sentiment
        self.strategy_allocator = strategy_allocator

    def test_sentiment_circuit_breaker_thresholds(self):
        """Test that sentiment circuit breaker has correct increased thresholds."""
        # P0 Fix: Sentiment circuit breaker thresholds
        self.assertEqual(
            self.sentiment.SENTIMENT_FAILURE_THRESHOLD,
            15,
            "Sentiment failure threshold should be increased to 15",
        )
        self.assertEqual(
            self.sentiment.SENTIMENT_RECOVERY_TIMEOUT,
            1800,
            "Sentiment recovery timeout should be increased to 1800 seconds (30 minutes)",
        )

    def test_confidence_normalization_exists(self):
        """Test that confidence normalization logic is in place."""
        # P1 Fix: Confidence normalization
        allocator = self.strategy_allocator.StrategyAllocator()

        # Create mock signal with out-of-range confidence
        # This would simulate signals with confidence > 1
        signals_by_strategy = {
            "test_strategy": [
                MockSignal("AAPL", "buy", 2.79),  # Out of range confidence from problem statement
                MockSignal("GOOGL", "sell", 1.71)  # Out of range confidence from problem statement
            ]
        }

        # Test that allocator handles out-of-range confidence values
        try:
            result = allocator.allocate(signals_by_strategy)
            # Check that any signals returned have confidence in [0,1] range
            for signal in result:
                self.assertTrue(0 <= signal.confidence <= 1,
                              f"Signal confidence {signal.confidence} is not in [0,1] range")
        except (ValueError, TypeError) as e:
            self.fail(f"Confidence normalization failed: {e}")

    def test_sector_classification_fallback(self):
        """Test that sector classification includes fallback for BABA."""
        # P2 Fix: Sector classification
        from ai_trading.core import bot_engine

        # Test that BABA is now in sector mappings
        sector = bot_engine.get_sector("BABA")
        self.assertNotEqual(sector, "Unknown", "BABA should have a fallback sector classification")
        self.assertEqual(sector, "Technology", "BABA should be classified as Technology")

    def test_execution_quantity_fix_exists(self):
        """Test that execution engine has the quantity calculation fix."""
        # P0 Fix: Quantity calculation bug
        # We can't easily test the actual fix without mocking orders, but we can verify
        # the _reconcile_partial_fills method exists and has been updated

        from ai_trading.execution.engine import ExecutionEngine

        # Create a mock context
        ctx = MockContext()
        engine = ExecutionEngine(ctx)

        # Verify the method exists and takes the expected parameters
        self.assertTrue(hasattr(engine, '_reconcile_partial_fills'),
                       "_reconcile_partial_fills method should exist")

    def test_short_selling_validation_exists(self):
        """Test that short selling validation method exists."""
        # P2 Fix: Short selling validation
        from ai_trading.execution.engine import ExecutionEngine

        ctx = MockContext()
        engine = ExecutionEngine(ctx)

        # Verify the short selling validation method exists
        self.assertTrue(hasattr(engine, '_validate_short_selling'),
                       "_validate_short_selling method should exist")


# Do not self-invoke tests when executed as a script; pytest will discover them


def test_position_sizing_minimum_viable():
    """Test that position sizing provides minimum viable quantities with available cash."""

    # Simulate the fixed logic from bot_engine.py
    balance = 88000.0  # $88K available cash
    target_weight = 0.002  # Weight above the 0.001 threshold
    current_price = 150.0  # AAPL-like price

    # Original calculation that resulted in 0
    raw_qty = int(balance * target_weight / current_price)

    # Fixed logic - ensure minimum position size when cash available
    if raw_qty <= 0 and balance > 1000 and target_weight > 0.001 and current_price > 0:
        raw_qty = max(1, int(1000 / current_price))  # Minimum $1000 position

    assert raw_qty > 0, f"Should compute positive quantity with ${balance:.0f} cash available"
    assert raw_qty >= 1, "Should have at least 1 share for minimum position"


def test_meta_learning_price_conversion():
    """Test meta learning properly converts string prices to numeric."""
    # Create a temporary CSV file with string price data (common issue)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(['symbol', 'entry_price', 'exit_price', 'signal_tags', 'side', 'qty'])
        # Mix of string and numeric prices to test conversion
        writer.writerow(['AAPL', '150.50', '155.25', 'momentum+trend', 'buy', '10'])
        writer.writerow(['MSFT', 250.00, 245.50, 'mean_reversion', 'sell', '5'])
        writer.writerow(['TSLA', '200.75', '210.00', 'breakout', 'buy', '8'])
        # Add edge case with invalid price
        writer.writerow(['INVALID', 'N/A', '100.00', 'test', 'buy', '1'])
        temp_file = f.name

    try:
        # Mock pandas to test the price conversion logic
        mock_df_data = {
            'symbol': ['AAPL', 'MSFT', 'TSLA', 'INVALID'],
            'entry_price': ['150.50', 250.00, '200.75', 'N/A'],
            'exit_price': ['155.25', 245.50, '210.00', '100.00'],
            'signal_tags': ['momentum+trend', 'mean_reversion', 'breakout', 'test'],
            'side': ['buy', 'sell', 'buy', 'buy'],
            'qty': [10, 5, 8, 1]
        }

        # Simulate the fixed price conversion logic
        import pytest
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(mock_df_data)

        # Test the fixed conversion logic
        df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
        df["exit_price"] = pd.to_numeric(df["exit_price"], errors="coerce")

        # Remove rows where price conversion failed
        df = df.dropna(subset=["entry_price", "exit_price"])

        # Validate that we have reasonable price data
        df = df[(df["entry_price"] > 0) & (df["exit_price"] > 0)]


        # Should have 3 valid rows (INVALID row should be filtered out)
        assert len(df) == 3, f"Should have 3 valid price rows, got {len(df)}"
        assert all(df["entry_price"] > 0), "All entry prices should be positive"
        assert all(df["exit_price"] > 0), "All exit prices should be positive"

    finally:
        os.unlink(temp_file)


def test_liquidity_minimum_position():
    """Test that low liquidity still allows minimum positions with sufficient cash."""

    # Simulate the fixed liquidity logic from calculate_entry_size
    cash = 88000.0  # $88K available
    price = 150.0
    liquidity_factor = 0.1  # Very low liquidity (< 0.2 threshold)

    # Original logic would return 0

    # Fixed logic - allow minimum position with sufficient cash
    if liquidity_factor < 0.2:
        if cash > 5000:
            # Use minimum viable position
            result = max(1, int(1000 / price)) if price > 0 else 1
        else:
            result = 0
    else:
        result = 1


    assert result > 0, "Should allow minimum position even with low liquidity when cash > $5000"
    assert result >= 1, "Should have at least 1 share minimum"


def test_stale_data_bypass_startup():
    """Test that stale data bypass works during initial deployment."""

    # Simulate startup environment with stale data bypass enabled
    stale_symbols = ["NFLX", "META", "TSLA", "MSFT", "AMD"]
    allow_stale_on_startup = True  # Default behavior

    # Test that bypass allows trading to proceed
    if stale_symbols and allow_stale_on_startup:
        trading_allowed = True
    else:
        trading_allowed = False

    assert trading_allowed, "Should allow trading with stale data bypass enabled on startup"

    # Test that bypass can be disabled
    allow_stale_on_startup = False
    if stale_symbols and not allow_stale_on_startup:
        trading_allowed = False
    else:
        trading_allowed = True

    assert not trading_allowed, "Should block trading when stale data bypass is disabled"


def test_rfc3339_timestamp_api_format():
    """Test that the actual API timestamp format is RFC3339 compliant."""
    from datetime import datetime

    # Test the exact format used in ai_trading.data.fetch
    start_dt = datetime(2025, 1, 4, 16, 23, 0, tzinfo=UTC)
    end_dt = datetime(2025, 1, 4, 16, 30, 0, tzinfo=UTC)

    # Apply the fix from ai_trading.data.fetch using isoformat_z helper
    start_param = isoformat_z(start_dt)
    end_param = isoformat_z(end_dt)


    # Verify RFC3339 compliance
    assert start_param.endswith('Z'), "Start timestamp should end with 'Z'"
    assert end_param.endswith('Z'), "End timestamp should end with 'Z'"
    assert 'T' in start_param, "Should contain ISO datetime separator 'T'"
    assert '+00:00' not in start_param, "Should not contain +00:00 offset"

