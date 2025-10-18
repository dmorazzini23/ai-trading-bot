"""Data Fetch Retry Handler with Exponential Backoff.

This module provides robust retry logic for data fetching operations
to handle transient failures and improve data quality.
"""

import logging
import time
from typing import Any, Callable, Optional, TypeVar
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 0.5,
        max_delay: float = 10.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            import random
            delay = delay * (0.5 + random.random())  # 50-150% of calculated delay
        
        return delay


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable] = None
):
    """
    Decorator to retry a function with exponential backoff.
    
    Args:
        config: Retry configuration
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback function called on each retry
    """
    
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    result = func(*args, **kwargs)
                    
                    # Log success after retry
                    if attempt > 0:
                        logger.info(
                            "RETRY_SUCCESS",
                            extra={
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "total_attempts": config.max_attempts
                            }
                        )
                    
                    return result
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < config.max_attempts - 1:
                        delay = config.get_delay(attempt)
                        
                        logger.warning(
                            "RETRY_ATTEMPT",
                            extra={
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "max_attempts": config.max_attempts,
                                "error": str(e),
                                "retry_delay": round(delay, 2)
                            }
                        )
                        
                        if on_retry:
                            on_retry(attempt, e)
                        
                        time.sleep(delay)
                    else:
                        logger.error(
                            "RETRY_EXHAUSTED",
                            extra={
                                "function": func.__name__,
                                "attempts": config.max_attempts,
                                "final_error": str(e)
                            }
                        )
            
            # All retries exhausted
            raise last_exception
        
        return wrapper
    return decorator


class DataQualityValidator:
    """Validates data quality and handles common issues."""
    
    @staticmethod
    def validate_ohlcv(df: Any) -> tuple[bool, str]:
        """
        Validate OHLCV dataframe quality.
        
        Returns:
            (is_valid, error_message) tuple
        """
        
        if df is None:
            return (False, "DataFrame is None")
        
        if len(df) == 0:
            return (False, "DataFrame is empty")
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            return (False, f"Missing columns: {missing_cols}")
        
        # Check for all NaN close prices
        if df['close'].isna().all():
            return (False, "All close prices are NaN")
        
        # Check for excessive NaN values (>50%)
        nan_pct = df['close'].isna().sum() / len(df) * 100
        if nan_pct > 50:
            return (False, f"Excessive NaN values: {nan_pct:.1f}%")
        
        # Check for non-positive prices
        if (df['close'] <= 0).any():
            return (False, "Contains non-positive prices")
        
        return (True, "Valid")
    
    @staticmethod
    def clean_ohlcv(df: Any) -> Any:
        """
        Clean OHLCV dataframe by handling common issues.
        
        Returns:
            Cleaned dataframe
        """
        
        if df is None or len(df) == 0:
            return df
        
        # Forward fill NaN values (use last known price)
        df = df.ffill()
        
        # Backward fill any remaining NaN at the start
        df = df.bfill()
        
        # Remove rows with non-positive prices
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df = df[df[col] > 0]
        
        return df
    
    @staticmethod
    def get_fallback_price(df: Any, symbol: str) -> Optional[float]:
        """
        Get a fallback price from dataframe when current price is unavailable.
        
        Returns:
            Last valid close price or None
        """
        
        if df is None or len(df) == 0:
            return None
        
        if 'close' not in df.columns:
            return None
        
        # Get last non-NaN close price
        valid_closes = df['close'].dropna()
        
        if len(valid_closes) == 0:
            return None
        
        price = float(valid_closes.iloc[-1])
        
        if price <= 0:
            return None
        
        logger.info(
            "FALLBACK_PRICE_USED",
            extra={
                "symbol": symbol,
                "price": price,
                "source": "last_valid_close"
            }
        )
        
        return price
