#!/usr/bin/env python3.12
"""
Performance optimizations for AI trading system.

Provides parallel processing, caching, and vectorized operations
for improved performance.
"""

import logging
import time
import functools
from typing import Dict, List, Optional, Callable, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import multiprocessing as mp

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Performance benchmark result."""
    operation: str
    duration_ms: float
    throughput_ops_per_sec: float
    memory_mb: float
    cpu_cores_used: int
    
    def __str__(self) -> str:
        return (f"{self.operation}: {self.duration_ms:.2f}ms, "
                f"{self.throughput_ops_per_sec:.0f} ops/sec, "
                f"{self.memory_mb:.1f}MB")


class PerformanceCache:
    """
    LRU cache for expensive operations with TTL support.
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """
        Initialize performance cache.
        
        Args:
            max_size: Maximum cache size
            ttl_seconds: Time-to-live for cache entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        if 'timestamp' not in entry:
            return True
        
        age = (datetime.now(timezone.utc) - entry['timestamp']).total_seconds()
        return age > self.ttl_seconds
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        expired_keys = [
            key for key, entry in self._cache.items()
            if self._is_expired(entry)
        ]
        
        for key in expired_keys:
            del self._cache[key]
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        if self._is_expired(entry):
            del self._cache[key]
            return None
        
        # Update access time (LRU)
        entry['last_accessed'] = datetime.now(timezone.utc)
        return entry['value']
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        # Cleanup if at capacity
        if len(self._cache) >= self.max_size:
            self._cleanup_expired()
            
            # If still at capacity, remove oldest
            if len(self._cache) >= self.max_size:
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].get('last_accessed', datetime.min)
                )
                del self._cache[oldest_key]
        
        now = datetime.now(timezone.utc)
        self._cache[key] = {
            'value': value,
            'timestamp': now,
            'last_accessed': now
        }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self.logger.info("Performance cache cleared")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds
        }


def cached_operation(
    cache_ttl: int = 300,
    cache_key_func: Optional[Callable] = None
):
    """
    Decorator for caching expensive operations.
    
    Args:
        cache_ttl: Cache TTL in seconds
        cache_key_func: Function to generate cache key
    """
    def decorator(func: Callable) -> Callable:
        cache = PerformanceCache(ttl_seconds=cache_ttl)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try cache first
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            
            return result
        
        wrapper.cache = cache
        return wrapper
    
    return decorator


class ParallelProcessor:
    """
    Parallel processing manager for CPU-bound operations.
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize parallel processor.
        
        Args:
            max_workers: Maximum worker processes (defaults to CPU count)
        """
        if max_workers is None:
            max_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid overhead
        
        self.max_workers = max_workers
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def parallel_apply(
        self,
        func: Callable,
        data_chunks: List[Any],
        *args,
        **kwargs
    ) -> List[Any]:
        """
        Apply function to data chunks in parallel.
        
        Args:
            func: Function to apply
            data_chunks: Data to process in parallel
            *args: Additional arguments to func
            **kwargs: Additional keyword arguments to func
            
        Returns:
            List of results from parallel execution
        """
        if len(data_chunks) <= 1 or self.max_workers <= 1:
            # Single-threaded fallback
            return [func(chunk, *args, **kwargs) for chunk in data_chunks]
        
        results = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(func, chunk, *args, **kwargs): i
                for i, chunk in enumerate(data_chunks)
            }
            
            # Collect results in order
            results = [None] * len(data_chunks)
            
            for future in as_completed(future_to_chunk):
                chunk_index = future_to_chunk[future]
                try:
                    result = future.result()
                    results[chunk_index] = result
                except Exception as e:
                    self.logger.error(f"Parallel task failed: {e}")
                    results[chunk_index] = None
        
        return results
    
    def parallel_indicators(
        self,
        price_data: pd.DataFrame,
        indicator_configs: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Calculate technical indicators in parallel.
        
        Args:
            price_data: Price data (OHLCV)
            indicator_configs: List of indicator configurations
            
        Returns:
            DataFrame with calculated indicators
        """
        if len(indicator_configs) == 0:
            return pd.DataFrame()
        
        # Split indicators into chunks for parallel processing
        chunk_size = max(1, len(indicator_configs) // self.max_workers)
        indicator_chunks = [
            indicator_configs[i:i + chunk_size]
            for i in range(0, len(indicator_configs), chunk_size)
        ]
        
        def calculate_indicator_chunk(configs: List[Dict[str, Any]]) -> pd.DataFrame:
            """Calculate indicators for a chunk of configs."""
            chunk_results = pd.DataFrame()
            
            for config in configs:
                try:
                    indicator_name = config['name']
                    indicator_func = config['function']
                    indicator_params = config.get('params', {})
                    
                    # Calculate indicator
                    result = indicator_func(price_data, **indicator_params)
                    
                    if isinstance(result, pd.Series):
                        chunk_results[indicator_name] = result
                    elif isinstance(result, pd.DataFrame):
                        # Multi-column indicator
                        for col in result.columns:
                            chunk_results[f"{indicator_name}_{col}"] = result[col]
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate {config.get('name', 'unknown')}: {e}")
            
            return chunk_results
        
        # Process chunks in parallel
        chunk_results = self.parallel_apply(calculate_indicator_chunk, indicator_chunks)
        
        # Combine results
        combined_result = pd.DataFrame()
        for chunk_result in chunk_results:
            if chunk_result is not None and not chunk_result.empty:
                combined_result = pd.concat([combined_result, chunk_result], axis=1)
        
        return combined_result


class VectorizedOperations:
    """
    Vectorized operations for improved pandas performance.
    """
    
    @staticmethod
    def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
        """
        Fast rolling z-score calculation.
        
        Args:
            series: Input series
            window: Rolling window size
            
        Returns:
            Rolling z-score series
        """
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        
        # Vectorized z-score calculation
        zscore = (series - rolling_mean) / rolling_std
        
        return zscore
    
    @staticmethod
    def rolling_correlation(
        x: pd.Series,
        y: pd.Series,
        window: int
    ) -> pd.Series:
        """
        Fast rolling correlation calculation.
        
        Args:
            x: First series
            y: Second series
            window: Rolling window size
            
        Returns:
            Rolling correlation series
        """
        # Align series
        aligned_data = pd.DataFrame({'x': x, 'y': y}).dropna()
        
        if len(aligned_data) < window:
            return pd.Series(dtype=float)
        
        return aligned_data['x'].rolling(window).corr(aligned_data['y'])
    
    @staticmethod
    def fast_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
        """
        Fast return calculation using vectorized operations.
        
        Args:
            prices: Price series
            periods: Number of periods for return calculation
            
        Returns:
            Return series
        """
        # Use numpy for faster calculation
        price_values = prices.values
        shifted_prices = np.roll(price_values, periods)
        
        # Calculate returns
        returns = (price_values - shifted_prices) / shifted_prices
        
        # Set initial values to NaN
        returns[:periods] = np.nan
        
        return pd.Series(returns, index=prices.index)
    
    @staticmethod
    def batch_technical_indicators(
        price_data: pd.DataFrame,
        window_sizes: List[int]
    ) -> pd.DataFrame:
        """
        Batch calculation of common technical indicators.
        
        Args:
            price_data: OHLCV data
            window_sizes: List of window sizes to calculate
            
        Returns:
            DataFrame with technical indicators
        """
        result = pd.DataFrame(index=price_data.index)
        
        # Vectorized calculations for each window
        for window in window_sizes:
            if window > len(price_data):
                continue
            
            # Moving averages
            result[f'sma_{window}'] = price_data['close'].rolling(window).mean()
            result[f'ema_{window}'] = price_data['close'].ewm(span=window).mean()
            
            # Volatility
            returns = VectorizedOperations.fast_returns(price_data['close'])
            result[f'vol_{window}'] = returns.rolling(window).std() * np.sqrt(252)
            
            # Momentum
            result[f'mom_{window}'] = price_data['close'].pct_change(window)
            
            # Bollinger Bands
            sma = result[f'sma_{window}']
            std = price_data['close'].rolling(window).std()
            result[f'bb_upper_{window}'] = sma + 2 * std
            result[f'bb_lower_{window}'] = sma - 2 * std
        
        return result


def benchmark_operation(
    operation_name: str,
    operation_func: Callable,
    *args,
    **kwargs
) -> BenchmarkResult:
    """
    Benchmark an operation's performance.
    
    Args:
        operation_name: Name of the operation
        operation_func: Function to benchmark
        *args: Arguments to the function
        **kwargs: Keyword arguments to the function
        
    Returns:
        BenchmarkResult with performance metrics
    """
    import psutil
    import gc
    
    # Force garbage collection before benchmark
    gc.collect()
    
    # Get initial memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run benchmark
    start_time = time.perf_counter()
    result = operation_func(*args, **kwargs)
    end_time = time.perf_counter()
    
    # Calculate metrics
    duration_ms = (end_time - start_time) * 1000
    
    # Estimate throughput (assume 1 operation if no obvious count)
    if hasattr(result, '__len__'):
        num_operations = len(result)
    elif isinstance(result, (pd.DataFrame, pd.Series)):
        num_operations = len(result)
    else:
        num_operations = 1
    
    throughput = num_operations / (duration_ms / 1000) if duration_ms > 0 else 0
    
    # Memory usage
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = final_memory - initial_memory
    
    # CPU cores (estimate based on system)
    cpu_cores = mp.cpu_count()
    
    return BenchmarkResult(
        operation=operation_name,
        duration_ms=duration_ms,
        throughput_ops_per_sec=throughput,
        memory_mb=memory_used,
        cpu_cores_used=cpu_cores
    )


# Global instances
_global_processor: Optional[ParallelProcessor] = None
_global_cache: Optional[PerformanceCache] = None


def get_parallel_processor() -> ParallelProcessor:
    """Get or create global parallel processor."""
    global _global_processor
    if _global_processor is None:
        _global_processor = ParallelProcessor()
    return _global_processor


def get_performance_cache() -> PerformanceCache:
    """Get or create global performance cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = PerformanceCache()
    return _global_cache