#!/usr/bin/env python3
"""
Focused test suite for the specific critical trading bot issues described in the problem statement.
"""

import os
import unittest
from unittest.mock import Mock, patch, MagicMock

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
            import sentiment
            
            # Problem statement requires 15 failures, not 5 or 8
            expected_failures = 15
            self.assertEqual(sentiment.SENTIMENT_FAILURE_THRESHOLD, expected_failures,
                           f"SENTIMENT_FAILURE_THRESHOLD should be {expected_failures}, got {sentiment.SENTIMENT_FAILURE_THRESHOLD}")
            
            # Problem statement requires 1800s (30 minutes), not 600s or 900s
            expected_recovery = 1800
            self.assertEqual(sentiment.SENTIMENT_RECOVERY_TIMEOUT, expected_recovery,
                           f"SENTIMENT_RECOVERY_TIMEOUT should be {expected_recovery}s, got {sentiment.SENTIMENT_RECOVERY_TIMEOUT}s")
            
            print("✓ Sentiment circuit breaker meets problem statement requirements")
            
        except ImportError as e:
            self.fail(f"Failed to import sentiment module: {e}")

    def test_meta_learning_minimum_trades_requirement(self):
        """Test that meta-learning minimum trade requirement is reduced to 3."""
        try:
            from bot_engine import load_global_signal_performance
            import inspect
            
            sig = inspect.signature(load_global_signal_performance)
            current_default = sig.parameters["min_trades"].default
            
            # Problem statement requires reducing from 10 to 3
            expected_min_trades = 3
            self.assertEqual(current_default, expected_min_trades,
                           f"min_trades default should be {expected_min_trades}, got {current_default}")
            
            print("✓ Meta-learning minimum trades meets problem statement requirements")
            
        except ImportError as e:
            self.fail(f"Failed to import bot_engine: {e}")
        except Exception as e:
            self.fail(f"Error checking min_trades: {e}")

    def test_pltr_sector_classification(self):
        """Test that PLTR is classified as Technology sector."""
        try:
            from bot_engine import get_sector
            
            # Problem statement requires PLTR to be classified as Technology
            pltr_sector = get_sector("PLTR")
            expected_sector = "Technology"
            
            self.assertEqual(pltr_sector, expected_sector,
                           f"PLTR should be classified as {expected_sector}, got {pltr_sector}")
            
            print("✓ PLTR sector classification meets problem statement requirements")
            
        except ImportError as e:
            self.fail(f"Failed to import bot_engine: {e}")
        except Exception as e:
            self.fail(f"Error checking PLTR sector: {e}")

    def test_order_quantity_tracking_clarity(self):
        """Test that order quantity tracking provides clear distinction between
        requested, submitted, and filled quantities."""
        # Check that the trade execution logs have clear field names
        trade_execution_path = "trade_execution.py"
        if os.path.exists(trade_execution_path):
            with open(trade_execution_path, 'r') as f:
                content = f.read()
                
                # Check for clear quantity field names in FULL_FILL_SUCCESS
                self.assertIn('"requested_qty":', content,
                            "FULL_FILL_SUCCESS should include clear requested_qty field")
                self.assertIn('"filled_qty":', content,
                            "FULL_FILL_SUCCESS should include clear filled_qty field")
                
                # Check for clear quantity field names in ORDER_FILL_CONSOLIDATED
                self.assertIn('"total_filled_qty":', content,
                            "ORDER_FILL_CONSOLIDATED should use clear total_filled_qty field name")
                
                print("✓ Order quantity tracking has clear field names")
        else:
            self.fail("trade_execution.py not found")


def run_problem_statement_tests():
    """Run the problem statement focused tests."""
    print("\n=== Problem Statement Requirements Test Suite ===")
    
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
    
    print("\n=== Test Results Summary ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_problem_statement_tests()
    exit(0 if success else 1)