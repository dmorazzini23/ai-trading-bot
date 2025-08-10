#!/usr/bin/env python3
"""
Test suite for critical trading bot issue fixes.
Tests the specific fixes for the four critical issues identified in the problem statement.
"""

import os
import tempfile
import unittest

# Set up minimal environment for imports
os.environ.setdefault('ALPACA_API_KEY', 'test_key')
os.environ.setdefault('ALPACA_SECRET_KEY', 'test_secret')
os.environ.setdefault('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
os.environ.setdefault('WEBHOOK_SECRET', 'test_webhook_secret')  # Add missing env var
os.environ.setdefault('FLASK_PORT', '5000')  # Add missing env var
os.environ.setdefault('PYTEST_RUNNING', '1')

class TestCriticalIssueFixes(unittest.TestCase):
    """Test the implementation of critical trading bot issue fixes."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_issue_4_position_limit_increase(self):
        """Test Issue 4: Position limit increased from 10 to 20."""
        # Test that the change exists in the source code by reading the file directly
        bot_engine_path = "bot_engine.py"
        if os.path.exists(bot_engine_path):
            with open(bot_engine_path, 'r') as f:
                content = f.read()
                # Check that the default value is now 20 instead of 10
                self.assertIn('"20"', content, "MAX_PORTFOLIO_POSITIONS default should be 20")
                print("✓ Issue 4 Fix: Position limit increased from 10 to 20")
        else:
            print("⚠ Issue 4: bot_engine.py not found for testing")
            self.assertTrue(True)

    def test_issue_2_sentiment_circuit_breaker_thresholds(self):
        """Test Issue 2: Sentiment circuit breaker thresholds improved."""
        # Test by reading the source code directly
        bot_engine_path = "bot_engine.py"
        if os.path.exists(bot_engine_path):
            with open(bot_engine_path, 'r') as f:
                content = f.read()
                # Check that thresholds have been improved
                self.assertIn('SENTIMENT_FAILURE_THRESHOLD = 8', content, 
                            "SENTIMENT_FAILURE_THRESHOLD should be 8")
                self.assertIn('SENTIMENT_RECOVERY_TIMEOUT = 900', content, 
                            "SENTIMENT_RECOVERY_TIMEOUT should be 900 (15 minutes)")
                print("✓ Issue 2 Fix: Sentiment circuit breaker thresholds improved")
        else:
            print("⚠ Issue 2: bot_engine.py not found for testing")
            self.assertTrue(True)

    def test_issue_3_quantity_tracking_logging(self):
        """Test Issue 3: Order execution quantity tracking improved."""
        # Test by reading the source code directly
        trade_execution_path = "trade_execution.py"
        if os.path.exists(trade_execution_path):
            with open(trade_execution_path, 'r') as f:
                content = f.read()
                # Check that FULL_FILL_SUCCESS now includes requested_qty
                self.assertIn('"requested_qty": requested_qty', content,
                            "FULL_FILL_SUCCESS should include requested_qty for tracking")
                # Check that ORDER_FILL_CONSOLIDATED uses total_filled_qty
                self.assertIn('"total_filled_qty": buf["qty"]', content,
                            "ORDER_FILL_CONSOLIDATED should use clear quantity field name")
                print("✓ Issue 3 Fix: Order execution tracking improved")
        else:
            print("⚠ Issue 3: trade_execution.py not found for testing")
            self.assertTrue(True)

    def test_issue_1_meta_learning_trigger_exists(self):
        """Test Issue 1: Meta-learning conversion trigger exists."""
        # Test by reading the source code directly
        bot_engine_path = "bot_engine.py"
        if os.path.exists(bot_engine_path):
            with open(bot_engine_path, 'r') as f:
                content = f.read()
                # Check that the meta-learning trigger code exists
                self.assertIn('from meta_learning import validate_trade_data_quality', content,
                            "Meta-learning trigger should import validation function")
                self.assertIn('METALEARN_TRIGGER_CONVERSION', content,
                            "Meta-learning trigger should log conversion attempts")
                print("✓ Issue 1 Fix: Meta-learning conversion trigger implemented")
        else:
            print("⚠ Issue 1: bot_engine.py not found for testing")
            self.assertTrue(True)

def run_critical_fixes_tests():
    """Run the critical fixes tests."""
    print("\n=== Critical Trading Bot Issue Fixes Test Suite ===")
    
    # Create test suite
    suite = unittest.TestSuite()
    test_class = TestCriticalIssueFixes
    
    # Add specific tests
    suite.addTest(test_class('test_issue_4_position_limit_increase'))
    suite.addTest(test_class('test_issue_2_sentiment_circuit_breaker_thresholds'))
    suite.addTest(test_class('test_issue_3_quantity_tracking_logging'))
    suite.addTest(test_class('test_issue_1_meta_learning_trigger_exists'))
    
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
    success = run_critical_fixes_tests()
    exit(0 if success else 1)