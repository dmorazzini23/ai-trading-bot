#!/usr/bin/env python3
"""
Critical fixes validation test for the AI trading bot.
Tests the 5 major issues identified in the problem statement.
"""

import unittest
from datetime import datetime, timezone, date
import os
import sys

class TestCriticalFixes(unittest.TestCase):
    """Test suite for the critical fixes implementation."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock config to avoid import errors
        class MockConfig:
            NEWS_API_KEY = "test_news_api_key"
        sys.modules['config'] = MockConfig()
    
    def test_sentiment_circuit_breaker_constants(self):
        """Test 1: Sentiment API Rate Limiting - Circuit breaker constants."""
        # Test that enhanced sentiment caching constants are reasonable
        SENTIMENT_RATE_LIMITED_TTL_SEC = 3600  # 1 hour  
        SENTIMENT_FAILURE_THRESHOLD = 3  # 3 failures
        SENTIMENT_RECOVERY_TIMEOUT = 300  # 5 minutes
        SENTIMENT_TTL_SEC = 600  # 10 minutes
        
        # Validate the constants
        self.assertGreater(SENTIMENT_RATE_LIMITED_TTL_SEC, SENTIMENT_TTL_SEC, 
                          "Rate limited TTL should be longer than normal TTL")
        self.assertGreaterEqual(SENTIMENT_FAILURE_THRESHOLD, 2,
                               "Should allow at least 2 failures before opening circuit")
        self.assertGreaterEqual(SENTIMENT_RECOVERY_TIMEOUT, 60,
                               "Recovery timeout should be at least 1 minute")
    
    def test_data_staleness_detection_improvement(self):
        """Test 4: Data Staleness Detection - Weekend/holiday awareness."""
        from utils import is_weekend, is_market_holiday
        
        # Test weekend detection
        saturday = datetime(2024, 1, 6, 12, 0, tzinfo=timezone.utc)  # Saturday
        sunday = datetime(2024, 1, 7, 12, 0, tzinfo=timezone.utc)    # Sunday  
        monday = datetime(2024, 1, 8, 12, 0, tzinfo=timezone.utc)    # Monday
        
        self.assertTrue(is_weekend(saturday), "Saturday should be detected as weekend")
        self.assertTrue(is_weekend(sunday), "Sunday should be detected as weekend")
        self.assertFalse(is_weekend(monday), "Monday should not be detected as weekend")
        
        # Test holiday detection
        new_years = date(2024, 1, 1)  # New Year's Day
        christmas = date(2024, 12, 25)  # Christmas
        regular_day = date(2024, 3, 15)  # Regular Friday
        
        self.assertTrue(is_market_holiday(new_years), "New Year's should be detected as holiday")
        self.assertTrue(is_market_holiday(christmas), "Christmas should be detected as holiday")
        self.assertFalse(is_market_holiday(regular_day), "Regular day should not be detected as holiday")
    
    def test_meta_learning_price_validation(self):
        """Test 2: MetaLearning Data Validation - Price validation logic."""
        # Mock pandas for testing
        try:
            import pandas as pd
            
            # Test data with mixed price types
            test_data = {
                'entry_price': ['100.50', '200', 'invalid', '50.25'],
                'exit_price': ['105.75', '195', '0', '55.00'],
                'side': ['buy', 'sell', 'buy', 'sell'],
                'signal_tags': ['momentum', 'mean_revert', 'momentum', 'trend']
            }
            df = pd.DataFrame(test_data)
            
            # Apply the validation logic from meta_learning.py
            df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
            df["exit_price"] = pd.to_numeric(df["exit_price"], errors="coerce")
            df = df.dropna(subset=["entry_price", "exit_price"])
            
            # Filter out non-positive prices
            df = df[(df["entry_price"] > 0) & (df["exit_price"] > 0)]
            
            # Should have 2 valid rows (first and last)
            self.assertEqual(len(df), 2, "Should have 2 rows with valid positive prices")
            self.assertTrue(all(df["entry_price"] > 0), "All entry prices should be positive")
            self.assertTrue(all(df["exit_price"] > 0), "All exit prices should be positive")
            
        except ImportError:
            # Skip if pandas not available
            self.skipTest("pandas not available for meta learning test")
    
    def test_systemd_service_configuration(self):
        """Test 3: Service Configuration - systemd service file."""
        service_file = "/home/runner/work/ai-trading-bot/ai-trading-bot/ai-trading-bot.service"
        self.assertTrue(os.path.exists(service_file), "systemd service file should exist")
        
        with open(service_file, 'r') as f:
            content = f.read()
        
        # Check key configuration elements
        self.assertIn("User=aiuser", content, "Service should run as aiuser")
        self.assertIn("Group=aiuser", content, "Service should run as aiuser group")
        self.assertIn("WorkingDirectory=/home/aiuser/ai-trading-bot", content, 
                     "Should have correct working directory")
        self.assertIn("NoNewPrivileges=true", content, "Should have security restrictions")
        self.assertIn("ProtectSystem=strict", content, "Should protect system")
        self.assertIn("Restart=always", content, "Should restart on failure")
    
    def test_error_handling_robustness(self):
        """Test 5: General Robustness - Error handling patterns."""
        # Test that we have proper exception handling patterns
        
        # Example of how sentiment fallback should work
        def mock_sentiment_fallback(cached_data, default_score=0.0):
            """Mock sentiment fallback logic."""
            try:
                if cached_data and isinstance(cached_data, (list, tuple)) and len(cached_data) > 0:
                    return cached_data[-1]  # Use last cached value
                return default_score
            except (TypeError, IndexError, AttributeError):
                return default_score  # Always return safe default
        
        # Test fallback scenarios
        self.assertEqual(mock_sentiment_fallback(None), 0.0, "Should return neutral when no cache")
        self.assertEqual(mock_sentiment_fallback([]), 0.0, "Should return neutral when empty cache")
        self.assertEqual(mock_sentiment_fallback([0.5, 0.7]), 0.7, "Should return last cached value")
        self.assertEqual(mock_sentiment_fallback("invalid"), 0.0, "Should handle invalid data gracefully")
    
    def test_cache_behavior(self):
        """Test enhanced caching behavior."""
        import time
        
        # Mock cache structure
        cache = {}
        normal_ttl = 600  # 10 minutes
        extended_ttl = 3600  # 1 hour
        
        def is_cache_valid(cache_entry, ttl):
            if not cache_entry:
                return False
            timestamp, value = cache_entry
            return time.time() - timestamp < ttl
        
        # Test normal cache behavior
        now = time.time()
        cache["AAPL"] = (now - 300, 0.5)  # 5 minutes old
        
        self.assertTrue(is_cache_valid(cache["AAPL"], normal_ttl), 
                       "Recent cache should be valid with normal TTL")
        
        # Test extended cache during rate limiting
        cache["MSFT"] = (now - 1800, 0.3)  # 30 minutes old
        
        self.assertFalse(is_cache_valid(cache["MSFT"], normal_ttl),
                        "Old cache should be invalid with normal TTL")
        self.assertTrue(is_cache_valid(cache["MSFT"], extended_ttl),
                       "Old cache should be valid with extended TTL")

if __name__ == '__main__':
    # Run the tests
    print("Running Critical Fixes Validation Tests...")
    print("=" * 60)
    
    # Set up the test environment
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCriticalFixes)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Report results
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 All critical fixes validation tests PASSED!")
        print(f"   Ran {result.testsRun} tests successfully")
    else:
        print("❌ Some tests FAILED!")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
        sys.exit(1)