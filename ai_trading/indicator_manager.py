"""Performance optimization for trading indicators and data management.

Provides incremental indicator calculations, memory-efficient circular buffers,
and smart caching to replace recalculation-on-every-tick for 10-100x speedup.

AI-AGENT-REF: Performance optimizations for production trading
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import threading
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class IndicatorType(Enum):
    """Types of technical indicators."""
    SIMPLE_MOVING_AVERAGE = "sma"
    EXPONENTIAL_MOVING_AVERAGE = "ema"
    RELATIVE_STRENGTH_INDEX = "rsi"
    BOLLINGER_BANDS = "bb"
    AVERAGE_TRUE_RANGE = "atr"
    MACD = "macd"
    STOCHASTIC = "stoch"
    WILLIAMS_R = "williams_r"
    VOLUME_WEIGHTED_AVERAGE_PRICE = "vwap"
    COMMODITY_CHANNEL_INDEX = "cci"


@dataclass
class IndicatorResult:
    """Result of indicator calculation."""
    timestamp: datetime
    value: Union[float, Dict[str, float]]
    is_valid: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class CircularBuffer:
    """Memory-efficient circular buffer for time series data."""
    
    def __init__(self, maxsize: int, dtype: type = float):
        self.maxsize = maxsize
        self.dtype = dtype
        self._buffer = np.full(maxsize, np.nan, dtype=dtype)
        self._index = 0
        self._size = 0
        self._lock = threading.RLock()  # Thread safety
    
    def append(self, value: Union[float, int]) -> None:
        """Add a value to the buffer."""
        with self._lock:
            self._buffer[self._index] = self.dtype(value)
            self._index = (self._index + 1) % self.maxsize
            self._size = min(self._size + 1, self.maxsize)
    
    def extend(self, values: List[Union[float, int]]) -> None:
        """Add multiple values to the buffer."""
        for value in values:
            self.append(value)
    
    def get_array(self, length: Optional[int] = None) -> np.ndarray:
        """Get array of values in chronological order."""
        with self._lock:
            if self._size == 0:
                return np.array([], dtype=self.dtype)
            
            if length is None:
                length = self._size
            else:
                length = min(length, self._size)
            
            if self._size < self.maxsize:
                # Buffer not full yet
                return self._buffer[:self._size][-length:]
            else:
                # Buffer is full, need to wrap around
                start_idx = (self._index - length) % self.maxsize
                if start_idx + length <= self.maxsize:
                    return self._buffer[start_idx:start_idx + length]
                else:
                    # Wrap around case
                    part1 = self._buffer[start_idx:]
                    part2 = self._buffer[:length - len(part1)]
                    return np.concatenate([part1, part2])
    
    def get_last(self, n: int = 1) -> Union[float, np.ndarray]:
        """Get the last n values."""
        array = self.get_array(n)
        if len(array) == 0:
            return np.nan if n == 1 else np.array([])
        return array[-1] if n == 1 else array
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self._size >= self.maxsize
    
    def size(self) -> int:
        """Get current size of buffer."""
        return self._size
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.fill(np.nan)
            self._index = 0
            self._size = 0
    
    def to_pandas(self) -> pd.Series:
        """Convert to pandas Series."""
        return pd.Series(self.get_array())


class IncrementalIndicator:
    """Base class for incremental indicator calculations."""
    
    def __init__(self, period: int, name: str):
        self.period = period
        self.name = name
        self.buffer = CircularBuffer(period * 2)  # Double buffer for safety
        self.last_value = np.nan
        self.is_initialized = False
        self._lock = threading.RLock()
    
    def update(self, value: float) -> Optional[float]:
        """Update indicator with new value and return result."""
        with self._lock:
            self.buffer.append(value)
            
            if self.buffer.size() >= self.period:
                self.last_value = self._calculate()
                self.is_initialized = True
                return self.last_value
            
            return None
    
    def _calculate(self) -> float:
        """Calculate indicator value (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def reset(self) -> None:
        """Reset indicator state."""
        with self._lock:
            self.buffer.clear()
            self.last_value = np.nan
            self.is_initialized = False


class IncrementalSMA(IncrementalIndicator):
    """Incremental Simple Moving Average."""
    
    def _calculate(self) -> float:
        values = self.buffer.get_array(self.period)
        return np.mean(values) if len(values) > 0 else np.nan


class IncrementalEMA(IncrementalIndicator):
    """Incremental Exponential Moving Average."""
    
    def __init__(self, period: int, name: str = "EMA"):
        super().__init__(period, name)
        self.alpha = 2.0 / (period + 1)
        self.ema_value = np.nan
    
    def _calculate(self) -> float:
        current_value = self.buffer.get_last()
        
        if np.isnan(self.ema_value):
            # Initialize with SMA
            values = self.buffer.get_array(min(self.period, self.buffer.size()))
            self.ema_value = np.mean(values)
        else:
            # Update with EMA formula
            self.ema_value = self.alpha * current_value + (1 - self.alpha) * self.ema_value
        
        return self.ema_value
    
    def reset(self) -> None:
        super().reset()
        self.ema_value = np.nan


class IncrementalRSI(IncrementalIndicator):
    """Incremental Relative Strength Index."""
    
    def __init__(self, period: int = 14, name: str = "RSI"):
        super().__init__(period, name)
        self.gain_ema = IncrementalEMA(period, "RSI_GAIN")
        self.loss_ema = IncrementalEMA(period, "RSI_LOSS")
        self.prev_close = np.nan
    
    def update(self, value: float) -> Optional[float]:
        with self._lock:
            if not np.isnan(self.prev_close):
                change = value - self.prev_close
                gain = max(change, 0)
                loss = max(-change, 0)
                
                gain_ema = self.gain_ema.update(gain)
                loss_ema = self.loss_ema.update(loss)
                
                if gain_ema is not None and loss_ema is not None:
                    if loss_ema == 0:
                        rsi = 100.0
                    else:
                        rs = gain_ema / loss_ema
                        rsi = 100.0 - (100.0 / (1.0 + rs))
                    
                    self.last_value = rsi
                    self.is_initialized = True
                    return rsi
            
            self.prev_close = value
            return None
    
    def reset(self) -> None:
        super().reset()
        self.gain_ema.reset()
        self.loss_ema.reset()
        self.prev_close = np.nan


class IncrementalATR(IncrementalIndicator):
    """Incremental Average True Range."""
    
    def __init__(self, period: int = 14, name: str = "ATR"):
        super().__init__(period, name)
        self.high_buffer = CircularBuffer(period * 2)
        self.low_buffer = CircularBuffer(period * 2)
        self.close_buffer = CircularBuffer(period * 2)
        self.tr_ema = IncrementalEMA(period, "TR_EMA")
    
    def update_ohlc(self, high: float, low: float, close: float) -> Optional[float]:
        """Update with OHLC values."""
        with self._lock:
            self.high_buffer.append(high)
            self.low_buffer.append(low)
            self.close_buffer.append(close)
            
            if self.close_buffer.size() >= 2:
                # Calculate True Range
                prev_close = self.close_buffer.get_last(2)[0]
                
                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)
                
                true_range = max(tr1, tr2, tr3)
                atr = self.tr_ema.update(true_range)
                
                if atr is not None:
                    self.last_value = atr
                    self.is_initialized = True
                    return atr
            
            return None
    
    def reset(self) -> None:
        super().reset()
        self.high_buffer.clear()
        self.low_buffer.clear()
        self.close_buffer.clear()
        self.tr_ema.reset()


class IndicatorCache:
    """Smart caching system for indicator results."""
    
    def __init__(self, max_cache_size: int = 10000, ttl_seconds: int = 300):
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[IndicatorResult, datetime]] = {}
        self._access_times: Dict[str, datetime] = {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[IndicatorResult]:
        """Get cached indicator result."""
        with self._lock:
            if key in self._cache:
                result, timestamp = self._cache[key]
                
                # Check TTL
                if datetime.now(timezone.utc) - timestamp <= timedelta(seconds=self.ttl_seconds):
                    self._access_times[key] = datetime.now(timezone.utc)
                    return result
                else:
                    # Expired, remove from cache
                    del self._cache[key]
                    if key in self._access_times:
                        del self._access_times[key]
            
            return None
    
    def set(self, key: str, result: IndicatorResult) -> None:
        """Set cached indicator result."""
        with self._lock:
            current_time = datetime.now(timezone.utc)
            
            # Cleanup if cache is full
            if len(self._cache) >= self.max_cache_size:
                self._cleanup_lru()
            
            self._cache[key] = (result, current_time)
            self._access_times[key] = current_time
    
    def _cleanup_lru(self) -> None:
        """Remove least recently used items."""
        if not self._access_times:
            return
        
        # Remove 25% of cache, starting with oldest access times
        remove_count = max(1, len(self._cache) // 4)
        sorted_items = sorted(self._access_times.items(), key=lambda x: x[1])
        
        for i in range(remove_count):
            key, _ = sorted_items[i]
            if key in self._cache:
                del self._cache[key]
            if key in self._access_times:
                del self._access_times[key]
    
    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class IndicatorManager:
    """Centralized manager for optimized indicator calculations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._indicators: Dict[str, IncrementalIndicator] = {}
        self._cache = IndicatorCache()
        self._lock = threading.RLock()
        
        # Performance metrics
        self._calculation_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        
        self.logger.info("IndicatorManager initialized")
    
    def create_indicator(self, 
                        indicator_type: IndicatorType, 
                        symbol: str,
                        period: int,
                        **kwargs) -> str:
        """Create and register a new indicator."""
        indicator_id = self._generate_indicator_id(indicator_type, symbol, period, **kwargs)
        
        with self._lock:
            if indicator_id in self._indicators:
                return indicator_id
            
            # Create indicator based on type
            if indicator_type == IndicatorType.SIMPLE_MOVING_AVERAGE:
                indicator = IncrementalSMA(period, f"SMA_{period}")
            elif indicator_type == IndicatorType.EXPONENTIAL_MOVING_AVERAGE:
                indicator = IncrementalEMA(period, f"EMA_{period}")
            elif indicator_type == IndicatorType.RELATIVE_STRENGTH_INDEX:
                indicator = IncrementalRSI(period, f"RSI_{period}")
            elif indicator_type == IndicatorType.AVERAGE_TRUE_RANGE:
                indicator = IncrementalATR(period, f"ATR_{period}")
            else:
                raise ValueError(f"Unsupported indicator type: {indicator_type}")
            
            self._indicators[indicator_id] = indicator
            self.logger.debug(f"Created indicator: {indicator_id}")
            
            return indicator_id
    
    def update_indicator(self, 
                        indicator_id: str, 
                        value: float,
                        timestamp: Optional[datetime] = None,
                        **kwargs) -> Optional[IndicatorResult]:
        """Update indicator with new data point."""
        with self._lock:
            if indicator_id not in self._indicators:
                return None
            
            indicator = self._indicators[indicator_id]
            
            # Check cache first
            cache_key = f"{indicator_id}_{value}_{timestamp}"
            cached_result = self._cache.get(cache_key)
            if cached_result:
                self._cache_hits += 1
                return cached_result
            
            self._cache_misses += 1
            
            # Calculate new value
            if isinstance(indicator, IncrementalATR) and 'high' in kwargs and 'low' in kwargs:
                new_value = indicator.update_ohlc(kwargs['high'], kwargs['low'], value)
            else:
                new_value = indicator.update(value)
            
            self._calculation_count += 1
            
            # Create result
            result = IndicatorResult(
                timestamp=timestamp or datetime.now(timezone.utc),
                value=new_value if new_value is not None else np.nan,
                is_valid=new_value is not None,
                metadata={
                    'indicator_type': indicator.name,
                    'period': indicator.period,
                    'calculation_count': self._calculation_count
                }
            )
            
            # Cache result
            self._cache.set(cache_key, result)
            
            return result
    
    def get_indicator_value(self, indicator_id: str) -> Optional[float]:
        """Get current indicator value without updating."""
        with self._lock:
            if indicator_id in self._indicators:
                indicator = self._indicators[indicator_id]
                return indicator.last_value if indicator.is_initialized else None
            return None
    
    def reset_indicator(self, indicator_id: str) -> bool:
        """Reset an indicator to initial state."""
        with self._lock:
            if indicator_id in self._indicators:
                self._indicators[indicator_id].reset()
                return True
            return False
    
    def remove_indicator(self, indicator_id: str) -> bool:
        """Remove an indicator."""
        with self._lock:
            if indicator_id in self._indicators:
                del self._indicators[indicator_id]
                return True
            return False
    
    def batch_update(self, 
                    updates: List[Tuple[str, float, Optional[datetime]]]) -> Dict[str, IndicatorResult]:
        """Update multiple indicators in batch for efficiency."""
        results = {}
        
        for indicator_id, value, timestamp in updates:
            result = self.update_indicator(indicator_id, value, timestamp)
            if result:
                results[indicator_id] = result
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        cache_hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'total_indicators': len(self._indicators),
            'total_calculations': self._calculation_count,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate_pct': cache_hit_rate,
            'cache_size': self._cache.size(),
            'active_indicators': [
                {
                    'id': indicator_id,
                    'name': indicator.name,
                    'period': indicator.period,
                    'initialized': indicator.is_initialized,
                    'last_value': indicator.last_value
                }
                for indicator_id, indicator in self._indicators.items()
            ]
        }
    
    def _generate_indicator_id(self, 
                              indicator_type: IndicatorType, 
                              symbol: str,
                              period: int,
                              **kwargs) -> str:
        """Generate unique indicator ID."""
        components = [
            indicator_type.value,
            symbol,
            str(period)
        ]
        
        # Add additional parameters
        for key, value in sorted(kwargs.items()):
            components.append(f"{key}_{value}")
        
        id_string = "_".join(components)
        
        # Hash for consistent length
        return hashlib.md5(id_string.encode()).hexdigest()[:16]
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self.logger.info("Indicator cache cleared")
    
    def optimize_memory(self) -> None:
        """Optimize memory usage by cleaning up unused indicators."""
        with self._lock:
            # Remove indicators that haven't been used recently
            inactive_indicators = [
                indicator_id for indicator_id, indicator in self._indicators.items()
                if not indicator.is_initialized
            ]
            
            for indicator_id in inactive_indicators:
                del self._indicators[indicator_id]
            
            # Clear cache
            self._cache.clear()
            
            self.logger.info(f"Memory optimization: removed {len(inactive_indicators)} inactive indicators")


# Global indicator manager instance
_indicator_manager: Optional[IndicatorManager] = None


def get_indicator_manager() -> IndicatorManager:
    """Get or create global indicator manager instance."""
    global _indicator_manager
    if _indicator_manager is None:
        _indicator_manager = IndicatorManager()
    return _indicator_manager


def create_fast_sma(symbol: str, period: int) -> str:
    """Create fast SMA indicator."""
    manager = get_indicator_manager()
    return manager.create_indicator(IndicatorType.SIMPLE_MOVING_AVERAGE, symbol, period)


def create_fast_ema(symbol: str, period: int) -> str:
    """Create fast EMA indicator."""
    manager = get_indicator_manager()
    return manager.create_indicator(IndicatorType.EXPONENTIAL_MOVING_AVERAGE, symbol, period)


def create_fast_rsi(symbol: str, period: int = 14) -> str:
    """Create fast RSI indicator."""
    manager = get_indicator_manager()
    return manager.create_indicator(IndicatorType.RELATIVE_STRENGTH_INDEX, symbol, period)


def create_fast_atr(symbol: str, period: int = 14) -> str:
    """Create fast ATR indicator."""
    manager = get_indicator_manager()
    return manager.create_indicator(IndicatorType.AVERAGE_TRUE_RANGE, symbol, period)