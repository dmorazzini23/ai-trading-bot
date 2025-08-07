"""
Exchange-aligned clock and scheduling module.

Provides timing synchronization with exchange schedules and validates
bar finality before signal generation.
"""

import logging
import time
from typing import Optional, Dict, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

try:
    import pandas_market_calendars as mcal
    MARKET_CALENDAR_AVAILABLE = True
except ImportError:
    MARKET_CALENDAR_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class BarValidation:
    """Result of bar finality validation."""
    symbol: str
    timeframe: str
    is_final: bool
    current_time: datetime
    bar_close_time: datetime
    skew_ms: float
    reason: Optional[str] = None


class AlignedClock:
    """
    Exchange-aligned clock with skew detection and bar validation.
    
    Ensures trading operations are synchronized with exchange time and
    validates that bars are final before generating signals.
    """
    
    def __init__(self, max_skew_ms: float = 250.0, exchange: str = "NYSE"):
        """
        Initialize aligned clock.
        
        Args:
            max_skew_ms: Maximum allowed time skew in milliseconds
            exchange: Exchange to align with (NYSE, NASDAQ, etc.)
        """
        self.max_skew_ms = max_skew_ms
        self.exchange = exchange
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize market calendar if available
        self.calendar = None
        if MARKET_CALENDAR_AVAILABLE:
            try:
                self.calendar = mcal.get_calendar(exchange)
            except Exception as e:
                self.logger.warning(f"Failed to load {exchange} calendar: {e}")
        
        # Cache for bar close times
        self._bar_close_cache: Dict[str, datetime] = {}
    
    def get_exchange_time(self) -> datetime:
        """
        Get current exchange time (EST/EDT for NYSE).
        
        Returns:
            Current time in exchange timezone
        """
        utc_now = datetime.now(timezone.utc)
        
        if self.calendar:
            try:
                # Get exchange timezone from calendar
                exchange_tz = self.calendar.tz
                return utc_now.astimezone(exchange_tz)
            except Exception as e:
                self.logger.warning(f"Failed to get exchange time: {e}")
        
        # Fallback to EST/EDT for NYSE
        try:
            import pytz
            if self.exchange in ("NYSE", "NASDAQ"):
                exchange_tz = pytz.timezone("America/New_York")
                return utc_now.astimezone(exchange_tz)
        except ImportError:
            pass
        
        # Final fallback to UTC
        return utc_now
    
    def next_bar_close(self, symbol: str, timeframe: str = "1m") -> datetime:
        """
        Calculate next bar close time for symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (1m, 5m, 15m, 1h, 1d)
            
        Returns:
            Next bar close time in exchange timezone
        """
        cache_key = f"{symbol}_{timeframe}"
        
        # Check cache first
        if cache_key in self._bar_close_cache:
            cached_close = self._bar_close_cache[cache_key]
            if cached_close > self.get_exchange_time():
                return cached_close
        
        current_time = self.get_exchange_time()
        
        # Parse timeframe
        interval_minutes = self._parse_timeframe_minutes(timeframe)
        
        if interval_minutes >= 1440:  # Daily or longer
            # Next market close
            next_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            if next_close <= current_time:
                next_close += timedelta(days=1)
        else:
            # Intraday - round up to next interval
            minutes_since_midnight = current_time.hour * 60 + current_time.minute
            next_interval = ((minutes_since_midnight // interval_minutes) + 1) * interval_minutes
            
            next_close = current_time.replace(
                hour=next_interval // 60,
                minute=next_interval % 60,
                second=0,
                microsecond=0
            )
            
            # Handle day rollover
            if next_close.day != current_time.day:
                next_close = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
                next_close += timedelta(days=1)
        
        # Skip weekends and holidays if calendar available
        if self.calendar:
            try:
                trading_days = self.calendar.valid_days(
                    start_date=next_close.date(),
                    end_date=next_close.date() + timedelta(days=7)
                )
                if len(trading_days) > 0:
                    next_trading_day = trading_days[0].date()
                    if next_close.date() != next_trading_day:
                        next_close = datetime.combine(next_trading_day, next_close.time())
                        next_close = next_close.replace(tzinfo=next_close.tzinfo)
            except Exception as e:
                self.logger.warning(f"Calendar check failed: {e}")
        
        # Cache the result
        self._bar_close_cache[cache_key] = next_close
        
        return next_close
    
    def _parse_timeframe_minutes(self, timeframe: str) -> int:
        """Parse timeframe string into minutes."""
        timeframe = timeframe.lower()
        
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 1440
        else:
            # Default to 1 minute
            return 1
    
    def check_skew(self, reference_time: Optional[datetime] = None) -> float:
        """
        Check time skew against reference time.
        
        Args:
            reference_time: Reference time to check against (defaults to system time)
            
        Returns:
            Skew in milliseconds (positive if local is ahead)
        """
        if reference_time is None:
            reference_time = datetime.now(timezone.utc)
        
        exchange_time = self.get_exchange_time()
        local_utc = datetime.now(timezone.utc)
        
        # Convert exchange time to UTC for comparison
        exchange_utc = exchange_time.astimezone(timezone.utc)
        
        skew_seconds = (local_utc - exchange_utc).total_seconds()
        skew_ms = skew_seconds * 1000
        
        if abs(skew_ms) > self.max_skew_ms:
            self.logger.warning(
                f"Time skew detected: {skew_ms:.1f}ms "
                f"(max allowed: {self.max_skew_ms}ms)"
            )
        
        return skew_ms
    
    def ensure_final_bar(self, symbol: str, timeframe: str = "1m") -> BarValidation:
        """
        Validate that the current bar is final before generating signals.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe to validate
            
        Returns:
            BarValidation result with finality status
        """
        current_time = self.get_exchange_time()
        next_close = self.next_bar_close(symbol, timeframe)
        
        # Calculate time until next bar close
        time_to_close = (next_close - current_time).total_seconds()
        
        # Check if we're too close to bar close (within skew tolerance)
        skew_ms = self.check_skew()
        buffer_seconds = (self.max_skew_ms + 100) / 1000  # Add 100ms buffer
        
        is_final = time_to_close > buffer_seconds
        
        validation = BarValidation(
            symbol=symbol,
            timeframe=timeframe,
            is_final=is_final,
            current_time=current_time,
            bar_close_time=next_close,
            skew_ms=skew_ms
        )
        
        if not is_final:
            validation.reason = (
                f"Too close to bar close: {time_to_close:.1f}s remaining "
                f"(need >{buffer_seconds:.1f}s buffer)"
            )
            
            self.logger.warning(
                f"Bar not final for {symbol} {timeframe}: {validation.reason}"
            )
        
        return validation
    
    def is_market_open(self, symbol: str, timestamp: Optional[datetime] = None) -> bool:
        """
        Check if market is open for trading.
        
        Args:
            symbol: Trading symbol
            timestamp: Time to check (defaults to now)
            
        Returns:
            True if market is open
        """
        if timestamp is None:
            timestamp = self.get_exchange_time()
        
        if not self.calendar:
            # Fallback - assume market hours 9:30 AM - 4:00 PM EST weekdays
            if timestamp.weekday() >= 5:  # Weekend
                return False
            
            market_open = timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = timestamp.replace(hour=16, minute=0, second=0, microsecond=0)
            
            return market_open <= timestamp <= market_close
        
        try:
            # Use calendar for accurate market hours
            schedule = self.calendar.schedule(
                start_date=timestamp.date(),
                end_date=timestamp.date()
            )
            
            if schedule.empty:
                return False
            
            market_open = schedule.iloc[0]['market_open']
            market_close = schedule.iloc[0]['market_close']
            
            # Convert to same timezone as timestamp
            market_open = market_open.tz_convert(timestamp.tzinfo)
            market_close = market_close.tz_convert(timestamp.tzinfo)
            
            return market_open <= timestamp <= market_close
            
        except Exception as e:
            self.logger.warning(f"Market hours check failed: {e}")
            return False
    
    def wait_for_aligned_tick(self, symbol: str, timeframe: str = "1m") -> BarValidation:
        """
        Wait until we're properly aligned for the next trading tick.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe to align with
            
        Returns:
            BarValidation when aligned
        """
        max_wait_seconds = 60  # Don't wait more than 1 minute
        start_time = time.time()
        
        while time.time() - start_time < max_wait_seconds:
            validation = self.ensure_final_bar(symbol, timeframe)
            
            if validation.is_final:
                return validation
            
            # Wait a bit before checking again
            time.sleep(0.1)
        
        # Return last validation even if not final
        self.logger.warning(
            f"Timed out waiting for aligned tick for {symbol} {timeframe}"
        )
        return validation


# Global aligned clock instance
_global_clock: Optional[AlignedClock] = None


def get_aligned_clock() -> AlignedClock:
    """Get or create global aligned clock instance."""
    global _global_clock
    if _global_clock is None:
        _global_clock = AlignedClock()
    return _global_clock


def ensure_final_bar(symbol: str, timeframe: str = "1m") -> BarValidation:
    """
    Convenience function to validate bar finality.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe to validate
        
    Returns:
        BarValidation result
    """
    clock = get_aligned_clock()
    return clock.ensure_final_bar(symbol, timeframe)


def is_market_open(symbol: str, timestamp: Optional[datetime] = None) -> bool:
    """
    Convenience function to check if market is open.
    
    Args:
        symbol: Trading symbol
        timestamp: Time to check (defaults to now)
        
    Returns:
        True if market is open
    """
    clock = get_aligned_clock()
    return clock.is_market_open(symbol, timestamp)