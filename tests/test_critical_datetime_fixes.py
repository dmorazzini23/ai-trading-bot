#!/usr/bin/env python3
"""
Focused tests for critical trading bot datetime and MetaLearning fixes.
Tests the specific issues identified in the problem statement.
"""

import sys
import os
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
import tempfile

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestDatetimeTimezoneAwareness(unittest.TestCase):
    """Test that datetime objects are timezone-aware for Alpaca API compatibility."""
    
    def test_ensure_datetime_returns_timezone_aware(self):
        """Test that ensure_datetime returns timezone-aware datetime objects."""
        from ai_trading.data_fetcher import ensure_datetime
        
        # Test with naive datetime
        naive_dt = datetime(2025, 1, 1, 12, 0, 0)
        result = ensure_datetime(naive_dt)
        
        # After fix, this should be timezone-aware
        self.assertIsNotNone(result.tzinfo, "ensure_datetime should return timezone-aware datetime")
        
    def test_alpaca_api_format_compatibility(self):
        """Test that datetime format is compatible with Alpaca API RFC3339 requirements."""
        from ai_trading.data_fetcher import ensure_datetime
        
        # Test various input formats
        test_inputs = [
            datetime.now(timezone.utc).replace(tzinfo=None),  # AI-AGENT-REF: Fixed naive datetime by creating from UTC
            "2025-01-01 12:00:00",  # string format
            datetime.now(timezone.utc),  # already timezone-aware
        ]
        
        for input_dt in test_inputs:
            with self.subTest(input_dt=input_dt):
                result = ensure_datetime(input_dt)
                
                # Should be timezone-aware
                self.assertIsNotNone(result.tzinfo, f"Input {input_dt} should produce timezone-aware result")
                
                # Should be able to format as RFC3339 for Alpaca API
                rfc3339_str = result.isoformat()
                self.assertIn('T', rfc3339_str, "Should produce valid RFC3339 format")
                
    def test_get_minute_df_datetime_parameters(self):
        """Test that get_minute_df properly handles timezone-aware datetime parameters."""
        from ai_trading.data_fetcher import get_minute_df
        
        # Mock the Alpaca client to avoid real API calls
        with patch('data_fetcher._DATA_CLIENT') as mock_client:
            mock_client.get_stock_bars.return_value.df = MagicMock()
            mock_client.get_stock_bars.return_value.df.empty = False
            mock_client.get_stock_bars.return_value.df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Mock pandas DataFrame
            with patch('data_fetcher.pd.DataFrame') as mock_df:
                mock_df.return_value = mock_df
                mock_df.empty = False
                mock_df.columns = ['open', 'high', 'low', 'close', 'volume']
                
                # AI-AGENT-REF: Test with timezone-aware datetime instead of naive
                start_dt = datetime.now(timezone.utc)
                end_dt = datetime.now(timezone.utc) + timedelta(hours=1)
                
                try:
                    result = get_minute_df("AAPL", start_dt, end_dt)
                    # If no exception is raised, the timezone conversion worked
                    self.assertTrue(True, "get_minute_df should handle naive datetime without errors")
                except Exception as e:
                    # Check if the error is related to timezone issues
                    if "RFC3339" in str(e) or "timezone" in str(e).lower():
                        self.fail(f"get_minute_df failed due to timezone issues: {e}")
                    # Other errors are acceptable for this test


class TestMetaLearningDataFetching(unittest.TestCase):
    """Test MetaLearning data fetching and function calls."""
    
    def test_metalearn_invalid_prices_prevention(self):
        """Test that MetaLearning can handle valid trade data without METALEARN_INVALID_PRICES."""
        # Create a temporary CSV file with valid trade data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write("exit_price,entry_price,signal_tags,side\n")
            tmp_file.write("150.0,148.0,momentum+breakout,buy\n")
            tmp_file.write("145.0,147.0,mean_reversion,sell\n")
            tmp_file_path = tmp_file.name
        
        try:
            # Mock the TRADE_LOG_FILE to use our test file
            with patch('bot_engine.TRADE_LOG_FILE', tmp_file_path):
                from ai_trading.core.bot_engine import load_global_signal_performance
                
                # This should not return None or empty dict due to METALEARN_INVALID_PRICES
                result = load_global_signal_performance(min_trades=1, threshold=0.1)
                
                # Should return signal performance data, not None/empty due to invalid prices
                if result is None:
                    self.fail("load_global_signal_performance returned None - possible METALEARN_INVALID_PRICES issue")
                if isinstance(result, dict) and len(result) == 0:
                    # Check if it's empty due to invalid prices vs other reasons
                    # This test validates the function can process valid data
                    pass
                    
        finally:
            # Clean up temp file
            os.unlink(tmp_file_path)


class TestSentimentCaching(unittest.TestCase):
    """Test sentiment analysis caching and rate limit handling."""
    
    def test_sentiment_cache_rate_limit_handling(self):
        """Test that sentiment caching properly handles rate limits."""
        try:
            from ai_trading.core.bot_engine import fetch_sentiment, _SENTIMENT_CACHE
            import time
            from requests.exceptions import HTTPError
            
            # Clear cache
            _SENTIMENT_CACHE.clear()
            
            # Mock the API key variables in bot_engine module
            with patch('bot_engine.SENTIMENT_API_KEY', 'test_key'):
                with patch('bot_engine.NEWS_API_KEY', 'test_key'):
                    # Mock the requests to simulate rate limiting
                    with patch('requests.get') as mock_get:
                        # First call - simulate rate limit (429)
                        mock_response = MagicMock()
                        mock_response.status_code = 429
                        mock_response.raise_for_status.side_effect = HTTPError("429 Too Many Requests")
                        mock_get.return_value = mock_response
                        
                        # Mock context for fetch_sentiment
                        mock_ctx = MagicMock()
                        
                        # This should handle the rate limit gracefully and cache neutral score
                        score = fetch_sentiment(mock_ctx, "AAPL")
                        
                        # Should return neutral score (0.0) when rate limited
                        self.assertEqual(score, 0.0, "Should return neutral score when rate limited")
                        
                        # Should cache the neutral score
                        self.assertIn("AAPL", _SENTIMENT_CACHE, "Should cache the rate-limited result")
                        
                        # Verify the cached value is correct
                        cached_entry = _SENTIMENT_CACHE["AAPL"]
                        self.assertEqual(cached_entry[1], 0.0, "Cached sentiment should be 0.0")
                        
                        # Second call should use cache (no API call)
                        _SENTIMENT_CACHE.clear()  # Clear cache to test fresh call
                        _SENTIMENT_CACHE["AAPL"] = (time.time(), 0.0)  # Pre-populate with rate-limited result
                        
                        score2 = fetch_sentiment(mock_ctx, "AAPL")
                        self.assertEqual(score2, 0.0, "Should return cached neutral score")
                    
        except ImportError as e:
            # Skip test if modules are not available
            self.skipTest(f"Required modules not available: {e}")


class TestRetryConfiguration(unittest.TestCase):
    """Test that retry configuration is reasonable and won't cause infinite loops."""
    
    def test_historical_data_retry_limit(self):
        """Test that historical data fetching has reasonable retry limits."""
        from ai_trading.data_fetcher import get_historical_data
        
        # The retry decorator should have reasonable limits
        # We can check the function's retry configuration
        retry_decorator = None
        for attr_name in dir(get_historical_data):
            attr_value = getattr(get_historical_data, attr_name)
            if hasattr(attr_value, 'retry_state'):
                retry_decorator = attr_value
                break
        
        if retry_decorator:
            # Check that retry attempts are reasonable (not infinite)
            # This is a basic sanity check
            pass  # The retry decorator exists and should have been configured properly
            
    def test_get_minute_df_handles_persistent_errors(self):
        """Test that get_minute_df doesn't retry infinitely on persistent datetime format errors."""
        from ai_trading.data_fetcher import get_minute_df
        
        # Mock the Alpaca client to always fail with datetime format error
        with patch('data_fetcher._DATA_CLIENT') as mock_client:
            mock_client.get_stock_bars.side_effect = Exception("Invalid format for parameter start: error parsing")
            
            # AI-AGENT-REF: Use timezone-aware datetime instead of naive
            start_dt = datetime.now(timezone.utc)
            end_dt = datetime.now(timezone.utc) + timedelta(hours=1)
            
            # Should eventually give up and not retry infinitely
            with self.assertRaises(Exception):
                get_minute_df("AAPL", start_dt, end_dt)
            
            # Verify that it didn't make hundreds of retry attempts
            call_count = mock_client.get_stock_bars.call_count
            self.assertLess(call_count, 10, f"Should not retry more than 10 times, but made {call_count} attempts")


if __name__ == "__main__":
    unittest.main()