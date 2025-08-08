"""Test minute cache helper functions for timestamp retrieval and age calculation."""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta
import pandas as pd


class TestMinuteCacheHelpers(unittest.TestCase):
    """Test minute cache timestamp and age helper functions."""

    def setUp(self):
        """Set up test with mock cache data."""
        # Create mock DataFrame
        self.mock_df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [100.5, 101.5, 102.5],
            'low': [99.5, 100.5, 101.5],
            'close': [100.2, 101.2, 102.2],
            'volume': [1000, 1100, 1200]
        })
        
        # Create test timestamps
        self.current_time = pd.Timestamp.now(tz="UTC")
        self.old_time = self.current_time - pd.Timedelta(minutes=10)
        
        # Mock cache with some symbols
        self.mock_cache = {
            "AAPL": (self.mock_df, self.current_time),
            "MSFT": (self.mock_df, self.old_time),
        }

    def test_get_cached_minute_timestamp_exists(self):
        """Test get_cached_minute_timestamp for existing symbol."""
        with patch('data_fetcher._MINUTE_CACHE', self.mock_cache):
            from data_fetcher import get_cached_minute_timestamp
            
            result = get_cached_minute_timestamp("AAPL")
            
            self.assertEqual(result, self.current_time)
            self.assertIsInstance(result, pd.Timestamp)

    def test_get_cached_minute_timestamp_not_exists(self):
        """Test get_cached_minute_timestamp for non-existing symbol."""
        with patch('data_fetcher._MINUTE_CACHE', self.mock_cache):
            from data_fetcher import get_cached_minute_timestamp
            
            result = get_cached_minute_timestamp("GOOGL")
            
            self.assertIsNone(result)

    def test_get_cached_minute_timestamp_empty_cache(self):
        """Test get_cached_minute_timestamp with empty cache."""
        with patch('data_fetcher._MINUTE_CACHE', {}):
            from data_fetcher import get_cached_minute_timestamp
            
            result = get_cached_minute_timestamp("AAPL")
            
            self.assertIsNone(result)

    def test_last_minute_bar_age_seconds_recent(self):
        """Test last_minute_bar_age_seconds for recently cached data."""
        with patch('data_fetcher._MINUTE_CACHE', self.mock_cache):
            with patch('pandas.Timestamp.now') as mock_now:
                # Mock current time to be 30 seconds after cache time
                mock_now.return_value = self.current_time + pd.Timedelta(seconds=30)
                
                from data_fetcher import last_minute_bar_age_seconds
                
                result = last_minute_bar_age_seconds("AAPL")
                
                self.assertAlmostEqual(result, 30.0, places=1)

    def test_last_minute_bar_age_seconds_old(self):
        """Test last_minute_bar_age_seconds for old cached data."""
        with patch('data_fetcher._MINUTE_CACHE', self.mock_cache):
            with patch('pandas.Timestamp.now') as mock_now:
                # Mock current time to be 5 minutes after old cache time
                mock_now.return_value = self.old_time + pd.Timedelta(minutes=5)
                
                from data_fetcher import last_minute_bar_age_seconds
                
                result = last_minute_bar_age_seconds("MSFT")
                
                self.assertAlmostEqual(result, 300.0, places=1)  # 5 minutes = 300 seconds

    def test_last_minute_bar_age_seconds_not_cached(self):
        """Test last_minute_bar_age_seconds for non-cached symbol."""
        with patch('data_fetcher._MINUTE_CACHE', self.mock_cache):
            from data_fetcher import last_minute_bar_age_seconds
            
            result = last_minute_bar_age_seconds("GOOGL")
            
            self.assertIsNone(result)

    def test_last_minute_bar_age_seconds_empty_cache(self):
        """Test last_minute_bar_age_seconds with empty cache."""
        with patch('data_fetcher._MINUTE_CACHE', {}):
            from data_fetcher import last_minute_bar_age_seconds
            
            result = last_minute_bar_age_seconds("AAPL")
            
            self.assertIsNone(result)

    def test_utc_handling(self):
        """Test that functions properly handle UTC timezone."""
        # Create timestamps with explicit UTC timezone
        utc_time = pd.Timestamp('2025-01-01 12:00:00', tz='UTC')
        cache_with_utc = {
            "TEST": (self.mock_df, utc_time)
        }
        
        with patch('data_fetcher._MINUTE_CACHE', cache_with_utc):
            from data_fetcher import get_cached_minute_timestamp
            
            result = get_cached_minute_timestamp("TEST")
            
            self.assertEqual(result.tz.zone, 'UTC')
            self.assertEqual(result, utc_time)

    def test_age_calculation_precision(self):
        """Test precision of age calculation."""
        base_time = pd.Timestamp('2025-01-01 12:00:00', tz='UTC')
        
        # Test various time differences
        test_cases = [
            (1, 1.0),      # 1 second
            (60, 60.0),    # 1 minute
            (90, 90.0),    # 1.5 minutes
            (300, 300.0),  # 5 minutes
            (3600, 3600.0) # 1 hour
        ]
        
        for seconds_diff, expected_age in test_cases:
            with self.subTest(seconds_diff=seconds_diff):
                cache_time = base_time
                current_time = base_time + pd.Timedelta(seconds=seconds_diff)
                
                cache = {"TEST": (self.mock_df, cache_time)}
                
                with patch('data_fetcher._MINUTE_CACHE', cache):
                    with patch('pandas.Timestamp.now', return_value=current_time):
                        from data_fetcher import last_minute_bar_age_seconds
                        
                        result = last_minute_bar_age_seconds("TEST")
                        
                        self.assertAlmostEqual(
                            result, 
                            expected_age, 
                            places=1,
                            msg=f"Expected {expected_age}s for {seconds_diff}s difference"
                        )

    def test_negative_age_handling(self):
        """Test handling of negative age (future timestamps)."""
        future_time = self.current_time + pd.Timedelta(minutes=5)
        cache_with_future = {
            "FUTURE": (self.mock_df, future_time)
        }
        
        with patch('data_fetcher._MINUTE_CACHE', cache_with_future):
            with patch('pandas.Timestamp.now', return_value=self.current_time):
                from data_fetcher import last_minute_bar_age_seconds
                
                result = last_minute_bar_age_seconds("FUTURE")
                
                # Should return negative value for future timestamp
                self.assertLess(result, 0)
                self.assertAlmostEqual(result, -300.0, places=1)  # -5 minutes

    def test_cache_structure_validation(self):
        """Test that functions handle malformed cache data gracefully."""
        malformed_cache = {
            "MALFORMED1": (self.mock_df,),  # Missing timestamp
            "MALFORMED2": ("not_a_df", self.current_time),  # Wrong data type
            "MALFORMED3": None,  # None value
        }
        
        with patch('data_fetcher._MINUTE_CACHE', malformed_cache):
            from data_fetcher import get_cached_minute_timestamp, last_minute_bar_age_seconds
            
            # Should handle malformed data gracefully
            for symbol in malformed_cache.keys():
                with self.subTest(symbol=symbol):
                    # Should not raise exception, might return None
                    try:
                        ts_result = get_cached_minute_timestamp(symbol)
                        age_result = last_minute_bar_age_seconds(symbol)
                        
                        # Results should be None for malformed data
                        if symbol == "MALFORMED1":
                            self.assertIsNone(ts_result)
                            self.assertIsNone(age_result)
                            
                    except (IndexError, TypeError, AttributeError):
                        # These exceptions are acceptable for malformed data
                        pass

    def test_multiple_symbols_consistency(self):
        """Test consistency when checking multiple symbols."""
        multi_symbol_cache = {
            "AAPL": (self.mock_df, self.current_time),
            "MSFT": (self.mock_df, self.old_time),
            "GOOGL": (self.mock_df, self.current_time - pd.Timedelta(minutes=2)),
        }
        
        with patch('data_fetcher._MINUTE_CACHE', multi_symbol_cache):
            with patch('pandas.Timestamp.now', return_value=self.current_time):
                from data_fetcher import get_cached_minute_timestamp, last_minute_bar_age_seconds
                
                # Get results for all symbols
                results = {}
                for symbol in multi_symbol_cache.keys():
                    results[symbol] = {
                        'timestamp': get_cached_minute_timestamp(symbol),
                        'age': last_minute_bar_age_seconds(symbol)
                    }
                
                # Verify results make sense relative to each other
                self.assertLess(
                    results["AAPL"]["age"],
                    results["MSFT"]["age"],
                    "AAPL should have smaller age than MSFT"
                )
                
                self.assertLess(
                    results["GOOGL"]["age"],
                    results["MSFT"]["age"],
                    "GOOGL should have smaller age than MSFT"
                )

    def test_units_are_seconds(self):
        """Test that age is returned in seconds (not minutes or other units)."""
        # Create cache with 1-minute old data
        one_minute_ago = self.current_time - pd.Timedelta(minutes=1)
        cache = {"TEST": (self.mock_df, one_minute_ago)}
        
        with patch('data_fetcher._MINUTE_CACHE', cache):
            with patch('pandas.Timestamp.now', return_value=self.current_time):
                from data_fetcher import last_minute_bar_age_seconds
                
                result = last_minute_bar_age_seconds("TEST")
                
                # Should be 60 seconds, not 1 (minute) or other unit
                self.assertAlmostEqual(result, 60.0, places=1)
                self.assertGreater(result, 50)  # Definitely in seconds range
                self.assertLess(result, 70)     # Not in minutes


if __name__ == "__main__":
    unittest.main()