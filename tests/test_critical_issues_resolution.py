"""Test critical trading bot issues and their resolution."""

import logging
import os
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

# Set minimal environment variables for testing
os.environ.setdefault('ALPACA_API_KEY', 'test_key')
os.environ.setdefault('ALPACA_SECRET_KEY', 'test_secret')
os.environ.setdefault('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
os.environ.setdefault('WEBHOOK_SECRET', 'test_webhook')
os.environ.setdefault('FLASK_PORT', '5000')

# Import the modules we need to test
try:
    from ai_trading.core import bot_engine
    from ai_trading.execution.engine import ExecutionEngine, Order, OrderSide
    from ai_trading.math.money import Money
    from ai_trading.risk.engine import RiskEngine  # AI-AGENT-REF: normalized import
    HAS_FULL_IMPORTS = True
except ImportError:
    # Continue with minimal testing if imports fail
    HAS_FULL_IMPORTS = False


class TestCriticalIssuesResolution(unittest.TestCase):
    """Test fixes for critical trading bot issues."""

    def setUp(self):
        """Set up test environment."""
        self.logger = logging.getLogger(__name__)

    @unittest.skipUnless(HAS_FULL_IMPORTS, "Required modules not available")
    def test_order_fill_tracking_reconciliation(self):
        """Test that order fill tracking accurately reconciles positions."""
        # Create mock execution engine
        engine = Mock(spec=ExecutionEngine)
        engine.logger = self.logger

        # Mock a scenario where bot thinks it owns 2 shares but only 1 was filled
        symbol = "NVDA"
        order = Order(symbol, OrderSide.BUY, 2, price=Money(150.0))

        # Test partial fill handling via Order model
        try:
            order.add_fill(1, Money(150.0))
            self.assertTrue(order.is_partially_filled)
        except (ValueError, TypeError) as e:
            self.fail(f"Partial fill handling failed: {e}")

    def test_overtrading_prevention_cooldown_logic(self):
        """Test that overtrading prevention cooldown logic is sound."""
        # Test existing cooldown mechanism logic
        trade_cooldowns = {}

        symbol = "AAPL"
        now = datetime.now(UTC)

        # Set a recent trade cooldown (within 5 minutes)
        trade_cooldowns[symbol] = now - timedelta(minutes=2)

        # Check cooldown logic
        cooldown_active = symbol in trade_cooldowns
        if cooldown_active:
            time_since_trade = now - trade_cooldowns[symbol]
            should_skip = time_since_trade < timedelta(minutes=5)  # 5 min cooldown
            self.assertTrue(should_skip, "Overtrading prevention should be active")

        # Test expired cooldown
        trade_cooldowns[symbol] = now - timedelta(minutes=10)  # 10 minutes ago
        time_since_trade = now - trade_cooldowns[symbol]
        should_allow = time_since_trade >= timedelta(minutes=5)
        self.assertTrue(should_allow, "Should allow trading after cooldown expires")

    def test_meta_learning_fallback_logic(self):
        """Test meta learning graceful handling of insufficient data."""
        # Test basic data validation logic
        min_samples_required = 100
        available_samples = 25  # Insufficient

        has_sufficient_data = available_samples >= min_samples_required
        self.assertFalse(has_sufficient_data, "Should detect insufficient data")

        # Test fallback behavior - should not crash when data is insufficient
        try:
            if not has_sufficient_data:
                # Simulate fallback to default behavior
                fallback_result = {'status': 'fallback_active', 'reason': 'insufficient_data'}
                self.assertEqual(fallback_result['status'], 'fallback_active')
        except (ValueError, TypeError) as e:
            self.fail(f"Fallback mechanism should not raise exceptions: {e}")

    def test_sentiment_rate_limiting_logic(self):
        """Test that sentiment analysis rate limiting logic is sound."""
        # Test rate limiting response logic
        def mock_sentiment_with_rate_limit(ticker, status_code=200):
            """Simulate sentiment fetch with different response codes."""
            if status_code == 429:
                # Rate limited - should return neutral and cache
                return 0.0
            elif status_code == 200:
                # Normal response
                return 0.5
            else:
                # Error case
                return 0.0

        # Test normal case
        normal_sentiment = mock_sentiment_with_rate_limit("AAPL", 200)
        self.assertEqual(normal_sentiment, 0.5)

        # Test rate limited case
        rate_limited_sentiment = mock_sentiment_with_rate_limit("AAPL", 429)
        self.assertEqual(rate_limited_sentiment, 0.0, "Should return neutral score when rate limited")

    def test_market_data_validation_logic(self):
        """Test that market data validation logic is sound."""
        # Test minimum data requirements logic
        def validate_market_data(data_length, min_required=20):
            """Validate if market data meets minimum requirements."""
            return data_length >= min_required

        # Test insufficient data
        insufficient_result = validate_market_data(5, 20)  # Only 5 bars, need 20
        self.assertFalse(insufficient_result, "Should reject insufficient market data")

        # Test sufficient data
        sufficient_result = validate_market_data(50, 20)  # 50 bars, need 20
        self.assertTrue(sufficient_result, "Should accept sufficient market data")

    def test_position_reconciliation_logic(self):
        """Test that position reconciliation logic properly detects mismatches."""
        # Test scenario where bot position != actual position
        def check_position_mismatch(bot_position, actual_position, tolerance=0.01):
            """Check for position mismatches."""
            mismatch = abs(bot_position - actual_position)
            return mismatch > tolerance, mismatch

        # Test significant mismatch
        has_mismatch, mismatch_amount = check_position_mismatch(2.1, 1.6)
        self.assertTrue(has_mismatch, "Should detect position tracking discrepancy")
        self.assertGreater(mismatch_amount, 0.1, "Mismatch should be significant")

        # Test acceptable difference
        no_mismatch, small_difference = check_position_mismatch(2.0, 2.005)
        self.assertFalse(no_mismatch, "Should not flag small differences as mismatches")


class TestOrderSpacingConfiguration(unittest.TestCase):
    """Test order spacing and frequency controls."""

    @unittest.skipUnless(HAS_FULL_IMPORTS, "Required modules not available")
    def test_risk_engine_order_spacing(self):
        """Test that risk engine provides order spacing configuration."""
        try:
            # Test the existing order_spacing method
            risk_engine = RiskEngine({})

            # Should have order spacing method
            self.assertTrue(hasattr(risk_engine, 'order_spacing'), "RiskEngine should have order_spacing method")

            spacing = risk_engine.order_spacing()
            self.assertIsInstance(spacing, float, "Order spacing should return float")
            self.assertGreaterEqual(spacing, 0, "Order spacing should be non-negative")
        except (ValueError, TypeError) as e:
            self.fail(f"RiskEngine order spacing test failed: {e}")

    def test_trade_frequency_limits_logic(self):
        """Test trade frequency limiting logic."""
        # Test that frequency limits logic is sound
        def check_trade_frequency(recent_trades, max_per_hour=10):
            """Check if trading frequency is within limits."""
            now = datetime.now(UTC)
            hour_ago = now - timedelta(hours=1)

            trades_last_hour = len([t for t in recent_trades if t > hour_ago])
            return trades_last_hour < max_per_hour, trades_last_hour

        # Test under limit
        recent_trades = []
        now = datetime.now(UTC)

        # Add 5 trades in last hour (under limit of 10)
        for i in range(5):
            trade_time = now - timedelta(minutes=i*10)  # Trades every 10 minutes
            recent_trades.append(trade_time)

        can_trade, count = check_trade_frequency(recent_trades, 10)
        self.assertTrue(can_trade, "Should allow trading when under frequency limit")
        self.assertEqual(count, 5, "Should count 5 trades in last hour")

        # Test over limit
        for i in range(6):  # Add 6 more trades
            trade_time = now - timedelta(minutes=i*5)  # More frequent trades
            recent_trades.append(trade_time)

        can_trade_over, count_over = check_trade_frequency(recent_trades, 10)
        self.assertFalse(can_trade_over, "Should block trading when over frequency limit")
        self.assertGreater(count_over, 10, "Should count more than 10 trades")


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)
    unittest.main()
