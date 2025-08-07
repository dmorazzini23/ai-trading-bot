"""
Per-symbol trading calendar registry for session validation.

Provides calendar-aware trading session validation to prevent trades
outside market hours and handle ETF half-days, futures sessions, etc.
"""

import logging
from datetime import datetime, timezone, time, date, timedelta
from typing import Dict, Optional, Tuple, Set, List
from dataclasses import dataclass
from enum import Enum
import pandas as pd

logger = logging.getLogger(__name__)


class AssetClass(Enum):
    """Asset class for calendar determination."""
    EQUITY = "equity"
    ETF = "etf" 
    FUTURES = "futures"
    FOREX = "forex"
    CRYPTO = "crypto"
    BOND = "bond"


@dataclass
class TradingSession:
    """Trading session definition."""
    name: str
    start_time: time  # Session start time (market timezone)
    end_time: time    # Session end time (market timezone)
    days_of_week: Set[int]  # 0=Monday, 6=Sunday
    timezone_name: str = "America/New_York"  # Market timezone
    
    def is_trading_day(self, dt: datetime) -> bool:
        """Check if given datetime falls on a trading day."""
        return dt.weekday() in self.days_of_week
    
    def is_in_session(self, dt: datetime) -> bool:
        """Check if given datetime is within trading session."""
        if not self.is_trading_day(dt):
            return False
        
        # Convert to market timezone
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        # For simplicity, assume UTC input and market timezone is EST/EDT
        # In production, would use proper timezone conversion
        market_time = dt.time()
        
        # Handle overnight sessions (e.g., futures)
        if self.start_time <= self.end_time:
            return self.start_time <= market_time <= self.end_time
        else:
            # Overnight session
            return market_time >= self.start_time or market_time <= self.end_time


class CalendarRegistry:
    """
    Registry of trading calendars by symbol and asset class.
    """
    
    def __init__(self):
        """Initialize calendar registry."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Symbol-specific calendars
        self._symbol_calendars: Dict[str, TradingSession] = {}
        
        # Asset class default calendars
        self._asset_calendars: Dict[AssetClass, TradingSession] = {}
        
        # Special trading days (holidays, half days)
        self._holidays: Set[date] = set()
        self._half_days: Set[date] = set()
        
        # Setup default calendars
        self._setup_default_calendars()
        self._setup_market_holidays()
    
    def _setup_default_calendars(self) -> None:
        """Setup default trading sessions by asset class."""
        # AI-AGENT-REF: Default trading sessions
        
        # US Equity regular session (9:30 AM - 4:00 PM ET)
        equity_session = TradingSession(
            name="US_EQUITY_REGULAR",
            start_time=time(9, 30),
            end_time=time(16, 0),
            days_of_week={0, 1, 2, 3, 4},  # Monday-Friday
            timezone_name="America/New_York"
        )
        
        # ETF session (same as equity for most ETFs)
        etf_session = TradingSession(
            name="US_ETF_REGULAR", 
            start_time=time(9, 30),
            end_time=time(16, 0),
            days_of_week={0, 1, 2, 3, 4},
            timezone_name="America/New_York"
        )
        
        # Futures extended session (6:00 PM - 5:00 PM ET next day)
        futures_session = TradingSession(
            name="US_FUTURES_EXTENDED",
            start_time=time(18, 0),  # 6 PM
            end_time=time(17, 0),    # 5 PM next day
            days_of_week={0, 1, 2, 3, 4, 6},  # Sunday 6PM - Friday 5PM
            timezone_name="America/New_York"
        )
        
        # Crypto 24/7
        crypto_session = TradingSession(
            name="CRYPTO_24_7",
            start_time=time(0, 0),
            end_time=time(23, 59),
            days_of_week={0, 1, 2, 3, 4, 5, 6},  # All days
            timezone_name="UTC"
        )
        
        # Forex extended (Sunday 5 PM - Friday 5 PM ET)
        forex_session = TradingSession(
            name="FOREX_EXTENDED",
            start_time=time(17, 0),  # Sunday 5 PM
            end_time=time(17, 0),    # Friday 5 PM
            days_of_week={0, 1, 2, 3, 4, 6},  # Sunday evening - Friday
            timezone_name="America/New_York"
        )
        
        # Bond market (9:00 AM - 3:00 PM ET)
        bond_session = TradingSession(
            name="US_BOND_REGULAR",
            start_time=time(9, 0),
            end_time=time(15, 0),
            days_of_week={0, 1, 2, 3, 4},
            timezone_name="America/New_York"
        )
        
        self._asset_calendars = {
            AssetClass.EQUITY: equity_session,
            AssetClass.ETF: etf_session,
            AssetClass.FUTURES: futures_session,
            AssetClass.CRYPTO: crypto_session,
            AssetClass.FOREX: forex_session,
            AssetClass.BOND: bond_session
        }
    
    def _setup_market_holidays(self) -> None:
        """Setup common US market holidays."""
        # AI-AGENT-REF: Common US market holidays for 2024-2025
        
        # 2024 holidays
        holidays_2024 = [
            date(2024, 1, 1),   # New Year's Day
            date(2024, 1, 15),  # MLK Day
            date(2024, 2, 19),  # Presidents Day
            date(2024, 3, 29),  # Good Friday
            date(2024, 5, 27),  # Memorial Day
            date(2024, 6, 19),  # Juneteenth
            date(2024, 7, 4),   # Independence Day
            date(2024, 9, 2),   # Labor Day
            date(2024, 11, 28), # Thanksgiving
            date(2024, 12, 25), # Christmas
        ]
        
        # 2025 holidays  
        holidays_2025 = [
            date(2025, 1, 1),   # New Year's Day
            date(2025, 1, 20),  # MLK Day
            date(2025, 2, 17),  # Presidents Day
            date(2025, 4, 18),  # Good Friday
            date(2025, 5, 26),  # Memorial Day
            date(2025, 6, 19),  # Juneteenth
            date(2025, 7, 4),   # Independence Day
            date(2025, 9, 1),   # Labor Day
            date(2025, 11, 27), # Thanksgiving
            date(2025, 12, 25), # Christmas
        ]
        
        self._holidays.update(holidays_2024 + holidays_2025)
        
        # Half days (early close at 1 PM ET)
        half_days = [
            date(2024, 7, 3),   # Day before July 4th
            date(2024, 11, 29), # Day after Thanksgiving
            date(2024, 12, 24), # Christmas Eve
            date(2025, 7, 3),   # Day before July 4th
            date(2025, 11, 28), # Day after Thanksgiving
            date(2025, 12, 24), # Christmas Eve
        ]
        
        self._half_days.update(half_days)
    
    def register_symbol(self, symbol: str, session: TradingSession) -> None:
        """
        Register custom trading session for a symbol.
        
        Args:
            symbol: Trading symbol
            session: Custom trading session
        """
        symbol = symbol.upper()
        self._symbol_calendars[symbol] = session
        self.logger.info(f"Registered custom calendar for {symbol}: {session.name}")
    
    def get_session(self, symbol: str, asset_class: Optional[AssetClass] = None) -> TradingSession:
        """
        Get trading session for symbol.
        
        Args:
            symbol: Trading symbol
            asset_class: Asset class if known (for classification)
            
        Returns:
            TradingSession for the symbol
        """
        symbol = symbol.upper()
        
        # Check for symbol-specific calendar first
        if symbol in self._symbol_calendars:
            return self._symbol_calendars[symbol]
        
        # Infer asset class if not provided
        if asset_class is None:
            asset_class = self._infer_asset_class(symbol)
        
        # Return asset class default
        return self._asset_calendars.get(asset_class, self._asset_calendars[AssetClass.EQUITY])
    
    def _infer_asset_class(self, symbol: str) -> AssetClass:
        """
        Infer asset class from symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Inferred asset class
        """
        symbol = symbol.upper()
        
        # Crypto patterns
        if any(crypto in symbol for crypto in ['BTC', 'ETH', 'USD', 'USDT']):
            if 'USD' in symbol and len(symbol) <= 6:
                return AssetClass.CRYPTO
        
        # Futures patterns (simplified)
        if symbol.endswith(('F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z')):
            if len(symbol) >= 3 and symbol[-2:].isdigit():
                return AssetClass.FUTURES
        
        # Common ETFs
        etf_symbols = {
            'SPY', 'QQQ', 'IWM', 'EFA', 'VTI', 'GLD', 'SLV', 'TLT',
            'AGG', 'LQD', 'HYG', 'XLF', 'XLK', 'XLE', 'XLI', 'XLV'
        }
        if symbol in etf_symbols:
            return AssetClass.ETF
        
        # Bond ETFs and treasury securities
        if any(bond in symbol for bond in ['AGG', 'TLT', 'LQD', 'HYG', 'BND']):
            return AssetClass.BOND
        
        # Default to equity
        return AssetClass.EQUITY
    
    def is_trading_day(self, symbol: str, dt: datetime) -> bool:
        """
        Check if given datetime is a trading day for symbol.
        
        Args:
            symbol: Trading symbol
            dt: Datetime to check
            
        Returns:
            True if trading day, False otherwise
        """
        # Check holidays
        check_date = dt.date() if isinstance(dt, datetime) else dt
        if check_date in self._holidays:
            return False
        
        # Check session trading days
        session = self.get_session(symbol)
        return session.is_trading_day(dt)
    
    def is_market_open(self, symbol: str, dt: datetime) -> bool:
        """
        Check if market is open for symbol at given time.
        
        Args:
            symbol: Trading symbol
            dt: Datetime to check
            
        Returns:
            True if market is open, False otherwise
        """
        if not self.is_trading_day(symbol, dt):
            return False
        
        session = self.get_session(symbol)
        is_in_session = session.is_in_session(dt)
        
        # Check for half days
        check_date = dt.date() if isinstance(dt, datetime) else dt.date()
        if check_date in self._half_days:
            # Market closes at 1 PM ET on half days
            market_time = dt.time()
            if market_time > time(13, 0):  # After 1 PM
                return False
        
        return is_in_session
    
    def get_session_bounds(
        self, 
        symbol: str, 
        trading_date: date
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Get session start and end times for a trading date.
        
        Args:
            symbol: Trading symbol
            trading_date: Trading date
            
        Returns:
            Tuple of (session_start, session_end) or (None, None) if not trading day
        """
        if not self.is_trading_day(symbol, datetime.combine(trading_date, time.min)):
            return None, None
        
        session = self.get_session(symbol)
        
        # Create datetime objects for session bounds
        session_start = datetime.combine(trading_date, session.start_time)
        session_end = datetime.combine(trading_date, session.end_time)
        
        # Handle overnight sessions
        if session.start_time > session.end_time:
            session_end += timedelta(days=1)
        
        # Adjust for half days
        if trading_date in self._half_days:
            session_end = datetime.combine(trading_date, time(13, 0))
        
        # Add timezone info (simplified - in production use proper timezone handling)
        session_start = session_start.replace(tzinfo=timezone.utc)
        session_end = session_end.replace(tzinfo=timezone.utc)
        
        return session_start, session_end
    
    def ensure_final_bar(self, symbol: str, timeframe: str) -> bool:
        """
        Check if we should ensure final bar for symbol/timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '1min', '5min', '1hour')
            
        Returns:
            True if final bar should be ensured
        """
        session = self.get_session(symbol)
        
        # For crypto and forex, final bar not needed (24/7 markets)
        if session.name in ['CRYPTO_24_7', 'FOREX_EXTENDED']:
            return False
        
        # For regular trading sessions, ensure final bar for intraday timeframes
        intraday_timeframes = {'1min', '5min', '15min', '30min', '1hour'}
        return timeframe in intraday_timeframes
    
    def get_next_trading_day(self, symbol: str, current_date: date) -> Optional[date]:
        """
        Get next trading day for symbol after current date.
        
        Args:
            symbol: Trading symbol
            current_date: Current date
            
        Returns:
            Next trading day or None if not found within reasonable period
        """
        check_date = current_date + timedelta(days=1)
        max_days = 10  # Reasonable search limit
        
        for _ in range(max_days):
            if self.is_trading_day(symbol, datetime.combine(check_date, time.min)):
                return check_date
            check_date += timedelta(days=1)
        
        return None
    
    def add_holiday(self, holiday_date: date) -> None:
        """Add a holiday to the calendar."""
        self._holidays.add(holiday_date)
        self.logger.info(f"Added holiday: {holiday_date}")
    
    def add_half_day(self, half_day_date: date) -> None:
        """Add a half day to the calendar."""
        self._half_days.add(half_day_date)
        self.logger.info(f"Added half day: {half_day_date}")


# Global calendar registry
_global_calendar: Optional[CalendarRegistry] = None


def get_calendar_registry() -> CalendarRegistry:
    """Get or create global calendar registry."""
    global _global_calendar
    if _global_calendar is None:
        _global_calendar = CalendarRegistry()
    return _global_calendar


# Convenience functions
def is_market_open(symbol: str, dt: Optional[datetime] = None) -> bool:
    """Check if market is open for symbol at given time (defaults to now)."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    
    calendar = get_calendar_registry()
    return calendar.is_market_open(symbol, dt)


def is_trading_day(symbol: str, dt: Optional[datetime] = None) -> bool:
    """Check if given datetime is a trading day for symbol (defaults to today)."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    
    calendar = get_calendar_registry()
    return calendar.is_trading_day(symbol, dt)


def ensure_final_bar(symbol: str, timeframe: str) -> bool:
    """Check if final bar should be ensured for symbol/timeframe."""
    calendar = get_calendar_registry()
    return calendar.ensure_final_bar(symbol, timeframe)