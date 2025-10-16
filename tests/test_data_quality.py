"""Tests for Data Quality Validator and Retry Handler."""

import pytest
import pandas as pd
import numpy as np
from ai_trading.data.retry_handler import DataQualityValidator, RetryConfig, retry_with_backoff


class TestDataQualityValidator:
    """Test data quality validation."""
    
    def test_validate_none_dataframe(self):
        """Test validation of None dataframe."""
        
        is_valid, msg = DataQualityValidator.validate_ohlcv(None)
        
        assert is_valid is False
        assert "None" in msg
    
    def test_validate_empty_dataframe(self):
        """Test validation of empty dataframe."""
        
        df = pd.DataFrame()
        is_valid, msg = DataQualityValidator.validate_ohlcv(df)
        
        assert is_valid is False
        assert "empty" in msg
    
    def test_validate_missing_columns(self):
        """Test validation with missing columns."""
        
        df = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            # Missing 'low', 'close', 'volume'
        })
        
        is_valid, msg = DataQualityValidator.validate_ohlcv(df)
        
        assert is_valid is False
        assert "Missing columns" in msg
    
    def test_validate_all_nan_close(self):
        """Test validation with all NaN close prices."""
        
        df = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [np.nan, np.nan],
            'volume': [1000, 2000]
        })
        
        is_valid, msg = DataQualityValidator.validate_ohlcv(df)
        
        assert is_valid is False
        assert "All close prices are NaN" in msg
    
    def test_validate_excessive_nan(self):
        """Test validation with excessive NaN values."""
        
        df = pd.DataFrame({
            'open': [100] * 10,
            'high': [102] * 10,
            'low': [99] * 10,
            'close': [np.nan] * 6 + [100] * 4,  # 60% NaN
            'volume': [1000] * 10
        })
        
        is_valid, msg = DataQualityValidator.validate_ohlcv(df)
        
        assert is_valid is False
        assert "Excessive NaN" in msg
    
    def test_validate_non_positive_prices(self):
        """Test validation with non-positive prices."""
        
        df = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [100, -50],  # Negative price
            'volume': [1000, 2000]
        })
        
        is_valid, msg = DataQualityValidator.validate_ohlcv(df)
        
        assert is_valid is False
        assert "non-positive" in msg
    
    def test_validate_valid_dataframe(self):
        """Test validation of valid dataframe."""
        
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [102, 103, 104],
            'low': [99, 100, 101],
            'close': [101, 102, 103],
            'volume': [1000, 2000, 3000]
        })
        
        is_valid, msg = DataQualityValidator.validate_ohlcv(df)
        
        assert is_valid is True
        assert msg == "Valid"
    
    def test_clean_ohlcv_forward_fill(self):
        """Test cleaning with forward fill."""
        
        df = pd.DataFrame({
            'open': [100, np.nan, 102],
            'high': [102, np.nan, 104],
            'low': [99, np.nan, 101],
            'close': [101, np.nan, 103],
            'volume': [1000, 2000, 3000]
        })
        
        cleaned = DataQualityValidator.clean_ohlcv(df)
        
        # NaN should be filled with previous value
        assert cleaned['close'].iloc[1] == 101
    
    def test_clean_ohlcv_remove_non_positive(self):
        """Test cleaning removes non-positive prices."""
        
        df = pd.DataFrame({
            'open': [100, 0, 102],
            'high': [102, 103, 104],
            'low': [99, 100, 101],
            'close': [101, -50, 103],
            'volume': [1000, 2000, 3000]
        })
        
        cleaned = DataQualityValidator.clean_ohlcv(df)
        
        # Row with non-positive price should be removed
        assert len(cleaned) == 2
        assert all(cleaned['close'] > 0)
    
    def test_get_fallback_price(self):
        """Test getting fallback price."""
        
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [102, 103, 104],
            'low': [99, 100, 101],
            'close': [101, 102, 103],
            'volume': [1000, 2000, 3000]
        })
        
        price = DataQualityValidator.get_fallback_price(df, "AAPL")
        
        assert price == 103  # Last close price
    
    def test_get_fallback_price_with_nan(self):
        """Test fallback price with NaN values."""
        
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [102, 103, 104],
            'low': [99, 100, 101],
            'close': [101, 102, np.nan],
            'volume': [1000, 2000, 3000]
        })
        
        price = DataQualityValidator.get_fallback_price(df, "AAPL")
        
        assert price == 102  # Last valid close price
    
    def test_get_fallback_price_none_df(self):
        """Test fallback price with None dataframe."""
        
        price = DataQualityValidator.get_fallback_price(None, "AAPL")
        
        assert price is None


class TestRetryConfig:
    """Test retry configuration."""
    
    def test_default_config(self):
        """Test default retry configuration."""
        
        config = RetryConfig()
        
        assert config.max_attempts == 3
        assert config.initial_delay == 0.5
        assert config.max_delay == 10.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
    
    def test_get_delay_exponential(self):
        """Test exponential delay calculation."""
        
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0, jitter=False)
        
        assert config.get_delay(0) == 1.0
        assert config.get_delay(1) == 2.0
        assert config.get_delay(2) == 4.0
        assert config.get_delay(3) == 8.0
    
    def test_get_delay_max_cap(self):
        """Test delay is capped at max_delay."""
        
        config = RetryConfig(initial_delay=1.0, max_delay=5.0, jitter=False)
        
        assert config.get_delay(10) == 5.0  # Would be 1024 without cap
    
    def test_get_delay_with_jitter(self):
        """Test delay with jitter."""
        
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0, jitter=True)
        
        delay = config.get_delay(1)
        
        # With jitter, delay should be between 1.0 and 3.0 (50-150% of 2.0)
        assert 1.0 <= delay <= 3.0


class TestRetryDecorator:
    """Test retry decorator."""
    
    def test_retry_success_first_attempt(self):
        """Test successful execution on first attempt."""
        
        call_count = 0
        
        @retry_with_backoff(config=RetryConfig(max_attempts=3))
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = successful_func()
        
        assert result == "success"
        assert call_count == 1
    
    def test_retry_success_after_failures(self):
        """Test successful execution after retries."""
        
        call_count = 0
        
        @retry_with_backoff(config=RetryConfig(max_attempts=3, initial_delay=0.1))
        def eventually_successful_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"
        
        result = eventually_successful_func()
        
        assert result == "success"
        assert call_count == 3
    
    def test_retry_exhausted(self):
        """Test retry exhaustion."""
        
        call_count = 0
        
        @retry_with_backoff(config=RetryConfig(max_attempts=3, initial_delay=0.1))
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            always_fails()
        
        assert call_count == 3
    
    def test_retry_specific_exceptions(self):
        """Test retry only on specific exceptions."""
        
        call_count = 0
        
        @retry_with_backoff(
            config=RetryConfig(max_attempts=3, initial_delay=0.1),
            exceptions=(ValueError,)
        )
        def specific_exception_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Retry this")
            elif call_count == 2:
                raise TypeError("Don't retry this")
        
        with pytest.raises(TypeError, match="Don't retry this"):
            specific_exception_func()
        
        assert call_count == 2  # Retried ValueError, but not TypeError


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

