"""
Exchange-aligned clock and scheduling module.

Provides timing synchronization with exchange schedules and validates
bar finality before signal generation.
"""

from ai_trading.logging import get_logger
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, tzinfo
from types import ModuleType
import importlib

mcal: ModuleType | None = None
MARKET_CALENDAR_AVAILABLE = False
logger = get_logger(__name__)


def _get_calendar(cal_name: str):
    global mcal, MARKET_CALENDAR_AVAILABLE
    if mcal is None:
        try:
            mcal = importlib.import_module("pandas_market_calendars")
            MARKET_CALENDAR_AVAILABLE = True
        except (ModuleNotFoundError, ValueError, TypeError) as exc:
            logger.warning(f"pandas_market_calendars not available: {exc}")
            return None
    try:
        return mcal.get_calendar(cal_name)  # type: ignore[union-attr]
    except (ValueError, TypeError) as exc:
        logger.warning(f"Failed to load {cal_name} calendar: {exc}")
        return None


@dataclass
class BarValidation:
    """Result of bar finality validation."""

    symbol: str
    timeframe: str
    is_final: bool
    current_time: datetime
    bar_close_time: datetime
    skew_ms: float
    reason: str | None = None


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
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.calendar = _get_calendar(exchange)
        self._bar_close_cache: dict[str, datetime] = {}

    def get_exchange_time(self, tz: tzinfo | None = None) -> datetime:
        """Get current exchange time.

        Args:
            tz: Optional timezone to convert the exchange time into.

        Returns:
            Current time in exchange timezone (or provided tz)
        """
        utc_now = datetime.now(UTC)
        current = utc_now
        if self.calendar:
            try:
                exchange_tz = self.calendar.tz
                current = utc_now.astimezone(exchange_tz)
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Failed to get exchange time: {e.__class__.__name__}: {e}")
        if tz is not None:
            current = current.astimezone(tz)
        return current

    def next_bar_close(self, symbol: str, timeframe: str = "1m", tz: tzinfo | None = None) -> datetime:
        """
        Calculate next bar close time for symbol and timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (1m, 5m, 15m, 1h, 1d)

        Returns:
            Next bar close time in exchange timezone
        """
        cache_key = f"{symbol}_{timeframe}_{str(tz) if tz else 'default'}"
        if cache_key in self._bar_close_cache:
            cached_close = self._bar_close_cache[cache_key]
            if cached_close > self.get_exchange_time(tz):
                return cached_close
        current_time = self.get_exchange_time(tz)
        interval_minutes = self._parse_timeframe_minutes(timeframe)
        if interval_minutes >= 1440:
            next_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            if next_close <= current_time:
                next_close += timedelta(days=1)
        else:
            minutes_since_midnight = current_time.hour * 60 + current_time.minute
            next_interval = (minutes_since_midnight // interval_minutes + 1) * interval_minutes
            next_close = current_time.replace(
                hour=next_interval // 60, minute=next_interval % 60, second=0, microsecond=0
            )
            if next_close.day != current_time.day:
                next_close = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
                next_close += timedelta(days=1)
        if self.calendar:
            try:
                trading_days = self.calendar.valid_days(
                    start_date=next_close.date(), end_date=next_close.date() + timedelta(days=7)
                )
                if len(trading_days) > 0:
                    next_trading_day = trading_days[0].date()
                    if next_close.date() != next_trading_day:
                        next_close = datetime.combine(
                            next_trading_day,
                            next_close.time(),
                            tzinfo=current_time.tzinfo,
                        )
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Calendar check failed: {e.__class__.__name__}: {e}")
        self._bar_close_cache[cache_key] = next_close
        return next_close

    def _parse_timeframe_minutes(self, timeframe: str) -> int:
        """Parse timeframe string into minutes."""
        timeframe = timeframe.lower()
        if timeframe.endswith("m"):
            return int(timeframe[:-1])
        elif timeframe.endswith("h"):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith("d"):
            return int(timeframe[:-1]) * 1440
        else:
            return 1

    def check_skew(self, reference_time: datetime | None = None, tz: tzinfo | None = None) -> float:
        """
        Check time skew against reference time.

        Args:
            reference_time: Reference time to check against (defaults to system time)

        Returns:
            Skew in milliseconds (positive if local is ahead)
        """
        if reference_time is None:
            local_utc = datetime.now(UTC)
        else:
            local_utc = reference_time.astimezone(UTC)
        exchange_utc = self.get_exchange_time(tz).astimezone(UTC)
        skew_seconds = (local_utc - exchange_utc).total_seconds()
        skew_ms = skew_seconds * 1000
        if abs(skew_ms) > self.max_skew_ms:
            self.logger.warning(f"Time skew detected: {skew_ms:.1f}ms (max allowed: {self.max_skew_ms}ms)")
        return skew_ms

    def ensure_final_bar(self, symbol: str, timeframe: str = "1m", tz: tzinfo | None = None) -> BarValidation:
        """
        Validate that the current bar is final before generating signals.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe to validate

        Returns:
            BarValidation result with finality status
        """
        current_time = self.get_exchange_time(tz)
        next_close = self.next_bar_close(symbol, timeframe, tz)
        time_to_close = (next_close - current_time).total_seconds()
        skew_ms = self.check_skew(tz=tz)
        buffer_seconds = (self.max_skew_ms + 100) / 1000
        is_final = time_to_close > buffer_seconds
        validation = BarValidation(
            symbol=symbol,
            timeframe=timeframe,
            is_final=is_final,
            current_time=current_time,
            bar_close_time=next_close,
            skew_ms=skew_ms,
        )
        if not is_final:
            validation.reason = (
                f"Too close to bar close: {time_to_close:.1f}s remaining (need >{buffer_seconds:.1f}s buffer)"
            )
            self.logger.warning(f"Bar not final for {symbol} {timeframe}: {validation.reason}")
        return validation

    def is_market_open(
        self, symbol: str, timestamp: datetime | None = None, tz: tzinfo | None = None
    ) -> bool:
        """
        Check if market is open for trading.

        Args:
            symbol: Trading symbol
            timestamp: Time to check (defaults to now)

        Returns:
            True if market is open
        """
        if timestamp is None:
            timestamp = self.get_exchange_time(tz)
        elif timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=tz or UTC)
        if not self.calendar:
            if timestamp.weekday() >= 5:
                return False
            market_open = timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = timestamp.replace(hour=16, minute=0, second=0, microsecond=0)
            return market_open <= timestamp <= market_close
        try:
            schedule = self.calendar.schedule(start_date=timestamp.date(), end_date=timestamp.date())
            if schedule.empty:
                return False
            market_open = schedule.iloc[0]["market_open"]
            market_close = schedule.iloc[0]["market_close"]
            market_open = market_open.tz_convert(timestamp.tzinfo)
            market_close = market_close.tz_convert(timestamp.tzinfo)
            return market_open <= timestamp <= market_close
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Market hours check failed: {e.__class__.__name__}: {e}")
            return False

    def wait_for_aligned_tick(
        self, symbol: str, timeframe: str = "1m", tz: tzinfo | None = None
    ) -> BarValidation:
        """
        Wait until we're properly aligned for the next trading tick.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe to align with

        Returns:
            BarValidation when aligned
        """
        max_wait_seconds = 60
        start_time = time.time()
        while time.time() - start_time < max_wait_seconds:
            validation = self.ensure_final_bar(symbol, timeframe, tz)
            if validation.is_final:
                return validation
            time.sleep(0.1)
        self.logger.warning(f"Timed out waiting for aligned tick for {symbol} {timeframe}")
        return validation


_global_clock: AlignedClock | None = None


def get_aligned_clock() -> AlignedClock:
    """Get or create global aligned clock instance."""
    global _global_clock
    if _global_clock is None:
        _global_clock = AlignedClock()
    return _global_clock


def ensure_final_bar(
    symbol: str, timeframe: str = "1m", tz: tzinfo | None = None
) -> BarValidation:
    """Convenience function to validate bar finality."""
    clock = get_aligned_clock()
    return clock.ensure_final_bar(symbol, timeframe, tz)


def is_market_open(
    symbol: str, timestamp: datetime | None = None, tz: tzinfo | None = None
) -> bool:
    """Convenience function to check if market is open."""
    clock = get_aligned_clock()
    return clock.is_market_open(symbol, timestamp, tz)
