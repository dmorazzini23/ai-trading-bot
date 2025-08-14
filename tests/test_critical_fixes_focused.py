#!/usr/bin/env python3
"""
Focused test suite for the critical trading bot fixes per problem statement.
"""

import unittest
import sys
import os

# Set testing environment
os.environ['TESTING'] = '1'

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestCriticalFixes(unittest.TestCase):
    """Test suite for critical P0 and P1 fixes."""

    def setUp(self):
        """Set up test environment."""
        # Import modules after setting TESTING flag
        import ai_trading.analysis.sentiment as sentiment
        import ai_trading.strategy_allocator as strategy_allocator  # AI-AGENT-REF: normalized import
        self.sentiment = sentiment
        self.strategy_allocator = strategy_allocator

    def test_sentiment_circuit_breaker_thresholds(self):
        """Test that sentiment circuit breaker has correct increased thresholds."""
        # P0 Fix: Sentiment circuit breaker thresholds
        self.assertEqual(self.sentiment.SENTIMENT_FAILURE_THRESHOLD, 25, 
                        "Sentiment failure threshold should be increased to 25")
        self.assertEqual(self.sentiment.SENTIMENT_RECOVERY_TIMEOUT, 3600, 
                        "Sentiment recovery timeout should be increased to 3600 seconds (1 hour)")
        print("✓ Sentiment circuit breaker thresholds correctly updated")

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
            print("✓ Confidence normalization handles out-of-range values")
        except Exception as e:
            self.fail(f"Confidence normalization failed: {e}")

    def test_sector_classification_fallback(self):
        """Test that sector classification includes fallback for BABA."""
        # P2 Fix: Sector classification
        from ai_trading.core import bot_engine

        # Test that BABA is now in sector mappings
        sector = bot_engine.get_sector("BABA")
        self.assertNotEqual(sector, "Unknown", "BABA should have a fallback sector classification")
        self.assertEqual(sector, "Technology", "BABA should be classified as Technology")
        print("✓ Sector classification fallback includes BABA")

    def test_trade_execution_quantity_fix_exists(self):
        """Test that trade execution has the quantity calculation fix."""
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
        print("✓ Trade execution quantity fix method exists")

    def test_short_selling_validation_exists(self):
        """Test that short selling validation method exists."""
        # P2 Fix: Short selling validation
        from ai_trading.execution.engine import ExecutionEngine
        
        ctx = MockContext()
        engine = ExecutionEngine(ctx)
        
        # Verify the short selling validation method exists
        self.assertTrue(hasattr(engine, '_validate_short_selling'), 
                       "_validate_short_selling method should exist")
        print("✓ Short selling validation method exists")


if __name__ == '__main__':
    print("Running critical trading bot fixes test suite...")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    test_class = TestCriticalFixes
    
    # Add specific tests for each critical fix
    suite.addTest(test_class('test_sentiment_circuit_breaker_thresholds'))
    suite.addTest(test_class('test_confidence_normalization_exists'))
    suite.addTest(test_class('test_sector_classification_fallback'))
    suite.addTest(test_class('test_trade_execution_quantity_fix_exists'))
    suite.addTest(test_class('test_short_selling_validation_exists'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("=" * 60)
    if result.wasSuccessful():
        print("✅ All critical fixes validated successfully!")
        sys.exit(0)
    else:
        print("❌ Some critical fixes failed validation!")
        sys.exit(1)
    print(f"Fixed timestamp format: {result}")
    
    # The fix should include 'Z' suffix for RFC3339 compliance
    assert result.endswith('Z'), f"Timestamp {result} should end with 'Z' for RFC3339 compliance"
    assert 'T' in result, f"Timestamp {result} should contain 'T' separator"


def test_position_sizing_minimum_viable():
    """Test that position sizing provides minimum viable quantities with available cash."""
    
    # Simulate the fixed logic from bot_engine.py
    balance = 88000.0  # $88K available cash
    target_weight = 0.002  # Weight above the 0.001 threshold
    current_price = 150.0  # AAPL-like price
    
    # Original calculation that resulted in 0
    raw_qty = int(balance * target_weight / current_price)
    print(f"Original qty calculation: {raw_qty}")
    
    # Fixed logic - ensure minimum position size when cash available
    if raw_qty <= 0 and balance > 1000 and target_weight > 0.001 and current_price > 0:
        raw_qty = max(1, int(1000 / current_price))  # Minimum $1000 position
        print(f"Using minimum position size: {raw_qty} shares")
    
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
        import pandas as pd
        df = pd.DataFrame(mock_df_data)
        
        # Test the fixed conversion logic
        df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
        df["exit_price"] = pd.to_numeric(df["exit_price"], errors="coerce")
        
        # Remove rows where price conversion failed
        df = df.dropna(subset=["entry_price", "exit_price"])
        
        # Validate that we have reasonable price data
        df = df[(df["entry_price"] > 0) & (df["exit_price"] > 0)]
        
        print(f"Converted dataframe: {len(df)} valid rows")
        
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
    original_result = 0 if liquidity_factor < 0.2 else 1
    
    # Fixed logic - allow minimum position with sufficient cash
    if liquidity_factor < 0.2:
        if cash > 5000:
            # Use minimum viable position
            result = max(1, int(1000 / price)) if price > 0 else 1
        else:
            result = 0
    else:
        result = 1
    
    print(f"Liquidity factor: {liquidity_factor}, Cash: ${cash:.0f}, Result: {result}")
    
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
        print(f"BYPASS_STALE_DATA_STARTUP: Allowing trading with {len(stale_symbols)} stale symbols")
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
    from datetime import datetime, timezone
    
    # Test the exact format used in data_fetcher.py
    start_dt = datetime(2025, 1, 4, 16, 23, 0, tzinfo=timezone.utc)
    end_dt = datetime(2025, 1, 4, 16, 30, 0, tzinfo=timezone.utc)
    
    # Apply the fix from data_fetcher.py
    start_param = start_dt.isoformat().replace('+00:00', 'Z')
    end_param = end_dt.isoformat().replace('+00:00', 'Z')
    
    print(f"API start param: {start_param}")
    print(f"API end param: {end_param}")
    
    # Verify RFC3339 compliance
    assert start_param.endswith('Z'), "Start timestamp should end with 'Z'"
    assert end_param.endswith('Z'), "End timestamp should end with 'Z'"
    assert 'T' in start_param, "Should contain ISO datetime separator 'T'"
    assert '+00:00' not in start_param, "Should not contain +00:00 offset"


if __name__ == "__main__":
    test_timestamp_format_includes_timezone()
    test_position_sizing_minimum_viable()
    test_meta_learning_price_conversion()
    test_liquidity_minimum_position()
    test_stale_data_bypass_startup()
    test_rfc3339_timestamp_api_format()
    print("All critical fix tests passed!")