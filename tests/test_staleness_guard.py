"""
Tests for data staleness guard functionality.
"""
import datetime
from unittest.mock import Mock
import pandas as pd

import pytest


class TestStalenessGuard:
    """Test data staleness validation functionality."""

    def test_staleness_guard_fresh_data(self):
        """Test staleness guard with fresh data."""
        # Import the function we're testing
        from ai_trading.core.bot_engine import _ensure_data_fresh
        
        # Create a mock fetcher that returns fresh data
        now = datetime.datetime.now(datetime.timezone.utc)
        fresh_timestamp = now - datetime.timedelta(seconds=30)  # 30 seconds old
        
        # Create test dataframe with fresh timestamp
        df = pd.DataFrame({
            'open': [100.0],
            'high': [101.0], 
            'low': [99.0],
            'close': [100.5],
            'volume': [1000],
            'timestamp': [fresh_timestamp]
        })
        
        # Mock fetcher
        mock_fetcher = Mock()
        mock_fetcher.get_minute_df = Mock(return_value=df)
        
        # Should not raise any exception for fresh data
        try:
            _ensure_data_fresh(mock_fetcher, ["AAPL"], max_age_seconds=300)
            success = True
        except Exception:
            success = False
            
        assert success, "Should not raise exception for fresh data"

    def test_staleness_guard_stale_data(self):
        """Test staleness guard with stale data."""
        from ai_trading.core.bot_engine import _ensure_data_fresh
        
        # Create a mock fetcher that returns stale data
        now = datetime.datetime.now(datetime.timezone.utc)
        stale_timestamp = now - datetime.timedelta(seconds=600)  # 10 minutes old
        
        # Create test dataframe with stale timestamp
        df = pd.DataFrame({
            'open': [100.0],
            'high': [101.0],
            'low': [99.0], 
            'close': [100.5],
            'volume': [1000],
            'timestamp': [stale_timestamp]
        })
        
        # Mock fetcher
        mock_fetcher = Mock()
        mock_fetcher.get_minute_df = Mock(return_value=df)
        
        # Should raise RuntimeError for stale data
        with pytest.raises(RuntimeError, match="Stale data for symbols"):
            _ensure_data_fresh(mock_fetcher, ["AAPL"], max_age_seconds=300)

    def test_staleness_guard_no_data(self):
        """Test staleness guard with no data."""
        from ai_trading.core.bot_engine import _ensure_data_fresh
        
        # Mock fetcher that returns None/empty data
        mock_fetcher = Mock()
        mock_fetcher.get_minute_df = Mock(return_value=None)
        
        # Should raise RuntimeError for no data
        with pytest.raises(RuntimeError, match="Stale data for symbols"):
            _ensure_data_fresh(mock_fetcher, ["AAPL"], max_age_seconds=300)

    def test_staleness_guard_empty_dataframe(self):
        """Test staleness guard with empty dataframe."""
        from ai_trading.core.bot_engine import _ensure_data_fresh
        
        # Mock fetcher that returns empty dataframe
        mock_fetcher = Mock()
        mock_fetcher.get_minute_df = Mock(return_value=pd.DataFrame())
        
        # Should raise RuntimeError for empty data
        with pytest.raises(RuntimeError, match="Stale data for symbols"):
            _ensure_data_fresh(mock_fetcher, ["AAPL"], max_age_seconds=300)

    def test_staleness_guard_multiple_symbols(self):
        """Test staleness guard with multiple symbols."""
        from ai_trading.core.bot_engine import _ensure_data_fresh
        
        now = datetime.datetime.now(datetime.timezone.utc)
        
        # Create mock fetcher that returns different data for different symbols
        def mock_get_minute_df(symbol, start, end):
            if symbol == "AAPL":
                # Fresh data for AAPL
                fresh_ts = now - datetime.timedelta(seconds=30)
                return pd.DataFrame({
                    'timestamp': [fresh_ts],
                    'close': [150.0]
                })
            elif symbol == "MSFT":
                # Stale data for MSFT  
                stale_ts = now - datetime.timedelta(seconds=600)
                return pd.DataFrame({
                    'timestamp': [stale_ts],
                    'close': [300.0]
                })
            else:
                # No data for other symbols
                return None
        
        mock_fetcher = Mock()
        mock_fetcher.get_minute_df = Mock(side_effect=mock_get_minute_df)
        
        # Should raise RuntimeError mentioning the stale symbol
        with pytest.raises(RuntimeError) as exc_info:
            _ensure_data_fresh(mock_fetcher, ["AAPL", "MSFT", "GOOGL"], max_age_seconds=300)
        
        error_msg = str(exc_info.value)
        assert "MSFT" in error_msg, "Error should mention MSFT as stale"
        assert "GOOGL" in error_msg, "Error should mention GOOGL as having no data"

    def test_staleness_guard_utc_logging(self):
        """Test that staleness guard logs UTC timestamps."""
        from ai_trading.core.bot_engine import _ensure_data_fresh
        from unittest.mock import patch
        
        # Mock logger to capture log messages
        with patch('ai_trading.core.bot_engine.logger') as mock_logger:
            now = datetime.datetime.now(datetime.timezone.utc)
            fresh_timestamp = now - datetime.timedelta(seconds=30)
            
            df = pd.DataFrame({
                'timestamp': [fresh_timestamp],
                'close': [100.0]
            })
            
            mock_fetcher = Mock()
            mock_fetcher.get_minute_df = Mock(return_value=df)
            
            # Call the function
            _ensure_data_fresh(mock_fetcher, ["AAPL"], max_age_seconds=300)
            
            # Verify debug log was called with UTC timestamp
            mock_logger.debug.assert_called()
            debug_call_args = mock_logger.debug.call_args[0]
            assert "UTC now=" in debug_call_args[0], "Should log UTC timestamp"
            
            # Verify the timestamp format is ISO
            assert "T" in debug_call_args[1], "Should use ISO format timestamp"

    def test_staleness_guard_timezone_handling(self):
        """Test staleness guard handles timezone-aware and naive timestamps."""
        from ai_trading.core.bot_engine import _ensure_data_fresh
        
        now = datetime.datetime.now(datetime.timezone.utc)
        
        # Test with timezone-naive timestamp (should be treated as UTC)
        naive_timestamp = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) - datetime.timedelta(seconds=30)  # AI-AGENT-REF: Create naive datetime from UTC
        df_naive = pd.DataFrame({
            'timestamp': [naive_timestamp],
            'close': [100.0]
        })
        
        # Test with timezone-aware timestamp  
        aware_timestamp = (now - datetime.timedelta(seconds=30)).replace(tzinfo=datetime.timezone.utc)
        df_aware = pd.DataFrame({
            'timestamp': [aware_timestamp], 
            'close': [100.0]
        })
        
        mock_fetcher = Mock()
        
        # Test both cases should work without error
        for df in [df_naive, df_aware]:
            mock_fetcher.get_minute_df = Mock(return_value=df)
            try:
                _ensure_data_fresh(mock_fetcher, ["AAPL"], max_age_seconds=300)
                success = True
            except Exception:
                success = False
            assert success, "Should handle both timezone-aware and naive timestamps"

    def test_staleness_guard_error_handling(self):
        """Test staleness guard handles fetcher errors gracefully."""
        from ai_trading.core.bot_engine import _ensure_data_fresh
        
        # Mock fetcher that raises an exception
        mock_fetcher = Mock()
        mock_fetcher.get_minute_df = Mock(side_effect=Exception("Network error"))
        
        # Should raise RuntimeError with error details
        with pytest.raises(RuntimeError) as exc_info:
            _ensure_data_fresh(mock_fetcher, ["AAPL"], max_age_seconds=300)
        
        error_msg = str(exc_info.value)
        assert "error=" in error_msg, "Should include error details in message"