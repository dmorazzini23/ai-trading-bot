#!/usr/bin/env python3
"""
Focused test suite for the specific critical trading bot issues described in the problem statement.
"""

import os
import sys
import unittest

# Set up minimal environment for imports
os.environ.setdefault('ALPACA_API_KEY', 'test_key')
os.environ.setdefault('ALPACA_SECRET_KEY', 'test_secret')
os.environ.setdefault('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
os.environ.setdefault('WEBHOOK_SECRET', 'test_webhook_secret')
os.environ.setdefault('FLASK_PORT', '5000')
os.environ.setdefault('PYTEST_RUNNING', '1')


class TestProblemStatementFixes(unittest.TestCase):
    """Test the specific fixes mentioned in the problem statement."""

    def test_sentiment_circuit_breaker_requirements(self):
        """Test that sentiment circuit breaker meets problem statement requirements:
        - Increase failure threshold to 15 (more tolerant)
        - Extend recovery timeout to 1800s (30 minutes)
        """
        try:
            from ai_trading.analysis import sentiment

            # Problem statement requires 15 failures, not 5 or 8
            expected_failures = 15
            self.assertEqual(sentiment.SENTIMENT_FAILURE_THRESHOLD, expected_failures,
                           f"SENTIMENT_FAILURE_THRESHOLD should be {expected_failures}, got {sentiment.SENTIMENT_FAILURE_THRESHOLD}")

            # Problem statement requires 1800s (30 minutes), not 600s or 900s
            expected_recovery = 1800
            self.assertEqual(sentiment.SENTIMENT_RECOVERY_TIMEOUT, expected_recovery,
                           f"SENTIMENT_RECOVERY_TIMEOUT should be {expected_recovery}s, got {sentiment.SENTIMENT_RECOVERY_TIMEOUT}s")


        except ImportError as e:
            self.fail(f"Failed to import sentiment module: {e}")

    def test_meta_learning_minimum_trades_requirement(self):
        """Test that meta-learning minimum trade requirement is reduced to 2."""
        # Test by reading the source code directly to avoid import issues
        bot_engine_path = "bot_engine.py"
        if os.path.exists(bot_engine_path):
            with open(bot_engine_path) as f:
                content = f.read()

            # Look for the environment variable default
            import re
            pattern = r'METALEARN_MIN_TRADES.*"(\d+)"'
            match = re.search(pattern, content)
            if match:
                current_value = int(match.group(1))
                expected_value = 2  # Updated from 3 to 2
                self.assertEqual(current_value, expected_value,
                               f"METALEARN_MIN_TRADES default should be {expected_value}, got {current_value}")
            else:
                self.fail("Could not find METALEARN_MIN_TRADES parameter in load_global_signal_performance")
        else:
            self.fail("bot_engine.py not found")

    def test_pltr_sector_classification(self):
        """Test that PLTR is classified as Technology sector."""
        # Test by reading the source code directly to avoid import issues
        bot_engine_path = "bot_engine.py"
        if os.path.exists(bot_engine_path):
            with open(bot_engine_path) as f:
                content = f.read()

            # Check if PLTR is in the Technology sector mapping
            if '"PLTR": "Technology"' in content:
                pass
            else:
                self.fail("PLTR not found in Technology sector mapping")
        else:
            self.fail("bot_engine.py not found")

    def test_order_quantity_tracking_clarity(self):
        """Test that order quantity tracking provides clear distinction between
        requested, submitted, and filled quantities."""
        # Check that the trade execution logs have clear field names
        from tests.support.mocks import MockContext
        from ai_trading.execution.engine import ExecutionEngine

        ctx = MockContext()
        engine = ExecutionEngine(ctx)

        with self.assertLogs('ai_trading.execution.engine', level='INFO') as log:
            engine._reconcile_partial_fills(
                requested_qty=10,
                remaining_qty=3,
                symbol='AAPL',
                side='buy',
            )
        output = ' '.join(log.output)
        self.assertIn('PARTIAL_FILL_DETECTED', output)
        self.assertIn('requested', output)
        self.assertIn('filled', output)


def run_problem_statement_tests():
    """Run the problem statement focused tests."""

    # Create test suite
    suite = unittest.TestSuite()
    test_class = TestProblemStatementFixes

    # Add specific tests based on problem statement
    suite.addTest(test_class('test_sentiment_circuit_breaker_requirements'))
    suite.addTest(test_class('test_meta_learning_minimum_trades_requirement'))
    suite.addTest(test_class('test_pltr_sector_classification'))
    suite.addTest(test_class('test_order_quantity_tracking_clarity'))

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)


    if result.failures:
        for test, traceback in result.failures:
            pass

    if result.errors:
        for test, traceback in result.errors:
            pass

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_problem_statement_tests()
    sys.exit(0 if success else 1)
