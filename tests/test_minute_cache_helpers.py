"""Test cache timestamp retrieval and age calculation."""

import pytest
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import patch

from data_fetcher import (
    get_cached_minute_timestamp, 
    last_minute_bar_age_seconds,
    _MINUTE_CACHE
)


def setup_function():
    """Clear cache before each test."""
    _MINUTE_CACHE.clear()


def test_get_cached_minute_timestamp_empty_cache():
    """Test get_cached_minute_timestamp with empty cache."""
    result = get_cached_minute_timestamp("AAPL")
    assert result is None


def test_get_cached_minute_timestamp_with_data():
    """Test get_cached_minute_timestamp with cached data."""
    # Create test data
    test_df = pd.DataFrame({
        'open': [100, 101],
        'high': [102, 103],
        'low': [99, 100],
        'close': [101, 102],
        'volume': [1000, 1100]
    })
    
    test_timestamp = pd.Timestamp.now(tz="UTC")
    _MINUTE_CACHE["AAPL"] = (test_df, test_timestamp)
    
    result = get_cached_minute_timestamp("AAPL")
    assert result == test_timestamp
    assert isinstance(result, pd.Timestamp)


def test_get_cached_minute_timestamp_invalid_timestamp():
    """Test get_cached_minute_timestamp with invalid timestamp type."""
    test_df = pd.DataFrame({'close': [100]})
    _MINUTE_CACHE["AAPL"] = (test_df, "not_a_timestamp")
    
    result = get_cached_minute_timestamp("AAPL")
    assert result is None


def test_last_minute_bar_age_seconds_empty_cache():
    """Test last_minute_bar_age_seconds with empty cache."""
    result = last_minute_bar_age_seconds("AAPL")
    assert result is None


def test_last_minute_bar_age_seconds_with_data():
    """Test last_minute_bar_age_seconds with cached data."""
    # Create test data with timestamp 60 seconds ago
    test_df = pd.DataFrame({'close': [100]})
    past_time = pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=60)
    _MINUTE_CACHE["AAPL"] = (test_df, past_time)
    
    result = last_minute_bar_age_seconds("AAPL")
    assert result is not None
    assert isinstance(result, int)
    # Should be approximately 60 seconds (allow some tolerance for test execution time)
    assert 58 <= result <= 62


def test_last_minute_bar_age_seconds_recent_data():
    """Test last_minute_bar_age_seconds with recent data."""
    # Create test data with current timestamp
    test_df = pd.DataFrame({'close': [100]})
    current_time = pd.Timestamp.now(tz="UTC")
    _MINUTE_CACHE["AAPL"] = (test_df, current_time)
    
    result = last_minute_bar_age_seconds("AAPL")
    assert result is not None
    assert isinstance(result, int)
    # Should be very small (0-2 seconds for test execution)
    assert 0 <= result <= 2


def test_last_minute_bar_age_seconds_no_timestamp():
    """Test last_minute_bar_age_seconds when get_cached_minute_timestamp returns None."""
    # This will trigger the case where get_cached_minute_timestamp returns None
    test_df = pd.DataFrame({'close': [100]})
    _MINUTE_CACHE["AAPL"] = (test_df, "invalid_timestamp")
    
    result = last_minute_bar_age_seconds("AAPL")
    assert result is None


def test_cache_helpers_integration():
    """Test that cache helpers work together correctly."""
    # Test with multiple symbols
    symbols = ["AAPL", "GOOGL", "MSFT"]
    current_time = pd.Timestamp.now(tz="UTC")
    
    for i, symbol in enumerate(symbols):
        test_df = pd.DataFrame({'close': [100 + i]})
        # Each symbol has data from a different time
        timestamp = current_time - pd.Timedelta(seconds=i * 30)
        _MINUTE_CACHE[symbol] = (test_df, timestamp)
    
    # Test timestamp retrieval
    for i, symbol in enumerate(symbols):
        ts = get_cached_minute_timestamp(symbol)
        assert ts is not None
        expected_time = current_time - pd.Timedelta(seconds=i * 30)
        # Allow some tolerance for timestamp comparison
        assert abs((ts - expected_time).total_seconds()) < 1
    
    # Test age calculation
    for i, symbol in enumerate(symbols):
        age = last_minute_bar_age_seconds(symbol)
        assert age is not None
        expected_age = i * 30
        # Allow some tolerance for execution time
        assert expected_age - 2 <= age <= expected_age + 2


def test_cache_helpers_timezone_handling():
    """Test that cache helpers properly handle timezone-aware timestamps."""
    # Test with different timezone representations
    test_df = pd.DataFrame({'close': [100]})
    
    # Test with UTC timestamp
    utc_time = pd.Timestamp.now(tz="UTC")
    _MINUTE_CACHE["AAPL"] = (test_df, utc_time)
    
    ts = get_cached_minute_timestamp("AAPL")
    assert ts is not None
    assert ts.tz is not None
    
    age = last_minute_bar_age_seconds("AAPL")
    assert age is not None
    assert age >= 0


def teardown_function():
    """Clear cache after each test."""
    _MINUTE_CACHE.clear()