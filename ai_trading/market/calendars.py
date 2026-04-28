"""
Per-symbol trading calendar registry for session validation.

Provides calendar-aware trading session validation to prevent trades
outside market hours and handle ETF half-days, futures sessions, etc.
"""
from ai_trading.logging import get_logger
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import Enum
import re
from zoneinfo import ZoneInfo
logger = get_logger(__name__)

_FOREX_CURRENCIES = {
    "AUD",
    "CAD",
    "CHF",
    "EUR",
    "GBP",
    "JPY",
    "NZD",
    "USD",
}
_FUTURES_SYMBOL_RE = re.compile(r"^/?[A-Z]{1,3}[FGHJKMNQUVXZ]\d{1,2}$")
_CRYPTO_BASES = {
    "AAVE",
    "ADA",
    "AVAX",
    "BCH",
    "BTC",
    "DOGE",
    "DOT",
    "ETH",
    "LINK",
    "LTC",
    "MATIC",
    "SHIB",
    "SOL",
    "UNI",
    "XRP",
}
_CRYPTO_QUOTES = {"BTC", "ETH", "EUR", "USD", "USDC", "USDT"}

def _observed_fixed_holiday(year: int, month: int, day: int) -> date:
    holiday = date(year, month, day)
    if holiday.weekday() == 5:
        return holiday - timedelta(days=1)
    if holiday.weekday() == 6:
        return holiday + timedelta(days=1)
    return holiday

def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    current = date(year, month, 1)
    offset = (weekday - current.weekday()) % 7
    return current + timedelta(days=offset + 7 * (n - 1))

def _last_weekday(year: int, month: int, weekday: int) -> date:
    current = date(year, month + 1, 1) - timedelta(days=1)
    offset = (current.weekday() - weekday) % 7
    return current - timedelta(days=offset)

def _good_friday(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day) - timedelta(days=2)

def _us_market_holidays(year: int) -> set[date]:
    holidays = {
        _observed_fixed_holiday(year, 1, 1),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _good_friday(year),
        _last_weekday(year, 5, 0),
        _observed_fixed_holiday(year, 6, 19),
        _observed_fixed_holiday(year, 7, 4),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed_fixed_holiday(year, 12, 25),
    }
    next_new_year = _observed_fixed_holiday(year + 1, 1, 1)
    if next_new_year.year == year:
        holidays.add(next_new_year)
    return holidays

def _us_market_half_days(year: int) -> set[date]:
    half_days = {_nth_weekday(year, 11, 3, 4) + timedelta(days=1)}
    christmas_eve = date(year, 12, 24)
    if christmas_eve.weekday() < 5:
        half_days.add(christmas_eve)
    july_third = date(year, 7, 3)
    if july_third.weekday() < 5:
        half_days.add(july_third)
    return half_days

def _looks_like_crypto_symbol(symbol: str) -> bool:
    compact = re.sub(r"[^A-Z0-9]", "", symbol.upper())
    for quote in sorted(_CRYPTO_QUOTES, key=len, reverse=True):
        if compact.endswith(quote):
            base = compact[: -len(quote)]
            if base in _CRYPTO_BASES:
                return True
    return False

class AssetClass(Enum):
    """Asset class for calendar determination."""
    EQUITY = 'equity'
    ETF = 'etf'
    FUTURES = 'futures'
    FOREX = 'forex'
    CRYPTO = 'crypto'
    BOND = 'bond'

@dataclass
class TradingSession:
    """Trading session definition."""
    name: str
    start_time: time
    end_time: time
    days_of_week: set[int]
    timezone_name: str = 'America/New_York'

    def _to_session_time(self, dt: datetime) -> datetime:
        """Convert a timestamp to the session's market timezone."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(ZoneInfo(self.timezone_name))

    def is_trading_day(self, dt: datetime | date) -> bool:
        """Check if given datetime falls on a trading day."""
        if isinstance(dt, datetime):
            if self.start_time >= self.end_time:
                return self.is_in_session(dt)
            dt = self._to_session_time(dt)
        return dt.weekday() in self.days_of_week

    def is_in_session(self, dt: datetime) -> bool:
        """Check if given datetime is within trading session."""
        market_dt = self._to_session_time(dt)
        market_time = market_dt.time()
        market_weekday = market_dt.weekday()
        if self.start_time == self.end_time:
            if market_time >= self.start_time:
                return market_weekday in self.days_of_week
            previous_weekday = (market_dt - timedelta(days=1)).weekday()
            return previous_weekday in self.days_of_week
        if self.start_time < self.end_time:
            if market_weekday not in self.days_of_week:
                return False
            return self.start_time <= market_time <= self.end_time
        if market_time >= self.start_time:
            return market_weekday in self.days_of_week
        if market_time < self.end_time:
            previous_weekday = (market_dt - timedelta(days=1)).weekday()
            return previous_weekday in self.days_of_week
        return False

class CalendarRegistry:
    """
    Registry of trading calendars by symbol and asset class.
    """

    def __init__(self):
        """Initialize calendar registry."""
        self.logger = get_logger(f'{__name__}.{self.__class__.__name__}')
        self._symbol_calendars: dict[str, TradingSession] = {}
        self._asset_calendars: dict[AssetClass, TradingSession] = {}
        self._holidays: set[date] = set()
        self._half_days: set[date] = set()
        self._holiday_years: set[int] = set()
        self._setup_default_calendars()
        self._setup_market_holidays()

    def _setup_default_calendars(self) -> None:
        """Setup default trading sessions by asset class."""
        equity_session = TradingSession(name='US_EQUITY_REGULAR', start_time=time(9, 30), end_time=time(16, 0), days_of_week={0, 1, 2, 3, 4}, timezone_name='America/New_York')
        etf_session = TradingSession(name='US_ETF_REGULAR', start_time=time(9, 30), end_time=time(16, 0), days_of_week={0, 1, 2, 3, 4}, timezone_name='America/New_York')
        futures_session = TradingSession(name='US_FUTURES_EXTENDED', start_time=time(18, 0), end_time=time(17, 0), days_of_week={0, 1, 2, 3, 6}, timezone_name='America/New_York')
        crypto_session = TradingSession(name='CRYPTO_24_7', start_time=time(0, 0), end_time=time(0, 0), days_of_week={0, 1, 2, 3, 4, 5, 6}, timezone_name='UTC')
        forex_session = TradingSession(name='FOREX_EXTENDED', start_time=time(17, 0), end_time=time(17, 0), days_of_week={0, 1, 2, 3, 6}, timezone_name='America/New_York')
        bond_session = TradingSession(name='US_BOND_REGULAR', start_time=time(9, 0), end_time=time(15, 0), days_of_week={0, 1, 2, 3, 4}, timezone_name='America/New_York')
        self._asset_calendars = {AssetClass.EQUITY: equity_session, AssetClass.ETF: etf_session, AssetClass.FUTURES: futures_session, AssetClass.CRYPTO: crypto_session, AssetClass.FOREX: forex_session, AssetClass.BOND: bond_session}

    def _setup_market_holidays(self) -> None:
        """Setup common US market holidays."""
        for year in range(2024, 2031):
            self._ensure_market_holidays(year)

    def _ensure_market_holidays(self, year: int) -> None:
        if year in self._holiday_years:
            return
        self._holidays.update(_us_market_holidays(year))
        self._half_days.update(_us_market_half_days(year))
        self._holiday_years.add(year)

    def register_symbol(self, symbol: str, session: TradingSession) -> None:
        """
        Register custom trading session for a symbol.

        Args:
            symbol: Trading symbol
            session: Custom trading session
        """
        symbol = symbol.upper()
        self._symbol_calendars[symbol] = session
        self.logger.info(f'Registered custom calendar for {symbol}: {session.name}')

    def get_session(self, symbol: str, asset_class: AssetClass | None=None) -> TradingSession:
        """
        Get trading session for symbol.

        Args:
            symbol: Trading symbol
            asset_class: Asset class if known (for classification)

        Returns:
            TradingSession for the symbol
        """
        symbol = symbol.upper()
        if symbol in self._symbol_calendars:
            return self._symbol_calendars[symbol]
        if asset_class is None:
            asset_class = self._infer_asset_class(symbol)
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
        if len(symbol) == 6 and symbol[:3] in _FOREX_CURRENCIES and symbol[3:] in _FOREX_CURRENCIES:
            return AssetClass.FOREX
        if _FUTURES_SYMBOL_RE.fullmatch(symbol):
            return AssetClass.FUTURES
        if _looks_like_crypto_symbol(symbol):
            return AssetClass.CRYPTO
        etf_symbols = {'SPY', 'QQQ', 'IWM', 'EFA', 'VTI', 'GLD', 'SLV', 'TLT', 'AGG', 'LQD', 'HYG', 'XLF', 'XLK', 'XLE', 'XLI', 'XLV'}
        if symbol in etf_symbols:
            return AssetClass.ETF
        if any((bond in symbol for bond in ['AGG', 'TLT', 'LQD', 'HYG', 'BND'])):
            return AssetClass.BOND
        return AssetClass.EQUITY

    def is_trading_day(self, symbol: str, dt: datetime | date) -> bool:
        """
        Check if given datetime is a trading day for symbol.

        Args:
            symbol: Trading symbol
            dt: Datetime to check

        Returns:
            True if trading day, False otherwise
        """
        session = self.get_session(symbol)
        if session.name in {'CRYPTO_24_7', 'FOREX_EXTENDED'}:
            return session.is_trading_day(dt)
        check_date = dt.date() if isinstance(dt, datetime) else dt
        self._ensure_market_holidays(check_date.year)
        if check_date in self._holidays:
            return False
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
        session = self.get_session(symbol)
        if session.name in {'CRYPTO_24_7', 'FOREX_EXTENDED', 'US_FUTURES_EXTENDED'}:
            return session.is_in_session(dt)
        if not self.is_trading_day(symbol, dt):
            return False
        is_in_session = session.is_in_session(dt)
        session_dt = session._to_session_time(dt)
        check_date = session_dt.date()
        self._ensure_market_holidays(check_date.year)
        if check_date in self._half_days:
            market_time = session_dt.time()
            if market_time > time(13, 0):
                return False
        return is_in_session

    def get_session_bounds(self, symbol: str, trading_date: date) -> tuple[datetime | None, datetime | None]:
        """
        Get session start and end times for a trading date.

        Args:
            symbol: Trading symbol
            trading_date: Trading date

        Returns:
            Tuple of (session_start, session_end) or (None, None) if not trading day
        """
        session = self.get_session(symbol)
        if session.start_time >= session.end_time:
            if trading_date.weekday() not in session.days_of_week:
                return (None, None)
        elif not self.is_trading_day(symbol, trading_date):
            return (None, None)
        self._ensure_market_holidays(trading_date.year)
        session_tz = ZoneInfo(session.timezone_name)
        session_start = datetime.combine(trading_date, session.start_time, tzinfo=session_tz)
        session_end = datetime.combine(trading_date, session.end_time, tzinfo=session_tz)
        if session.start_time >= session.end_time:
            session_end += timedelta(days=1)
        if session.name not in {'CRYPTO_24_7', 'FOREX_EXTENDED', 'US_FUTURES_EXTENDED'} and trading_date in self._half_days:
            session_end = datetime.combine(trading_date, time(13, 0), tzinfo=session_tz)
        session_start = session_start.astimezone(UTC)
        session_end = session_end.astimezone(UTC)
        return (session_start, session_end)

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
        if session.name in ['CRYPTO_24_7', 'FOREX_EXTENDED']:
            return False
        intraday_timeframes = {'1min', '5min', '15min', '30min', '1hour'}
        return timeframe in intraday_timeframes

    def get_next_trading_day(self, symbol: str, current_date: date) -> date | None:
        """
        Get next trading day for symbol after current date.

        Args:
            symbol: Trading symbol
            current_date: Current date

        Returns:
            Next trading day or None if not found within reasonable period
        """
        check_date = current_date + timedelta(days=1)
        max_days = 10
        for _ in range(max_days):
            if self.is_trading_day(symbol, check_date):
                return check_date
            check_date += timedelta(days=1)
        return None

    def add_holiday(self, holiday_date: date) -> None:
        """Add a holiday to the calendar."""
        self._holidays.add(holiday_date)
        self.logger.info(f'Added holiday: {holiday_date}')

    def add_half_day(self, half_day_date: date) -> None:
        """Add a half day to the calendar."""
        self._half_days.add(half_day_date)
        self.logger.info(f'Added half day: {half_day_date}')
_global_calendar: CalendarRegistry | None = None

def get_calendar_registry() -> CalendarRegistry:
    """Get or create global calendar registry."""
    global _global_calendar
    if _global_calendar is None:
        _global_calendar = CalendarRegistry()
    return _global_calendar

def is_market_open(symbol: str, dt: datetime | None=None) -> bool:
    """Check if market is open for symbol at given time (defaults to now)."""
    if dt is None:
        dt = datetime.now(UTC)
    calendar = get_calendar_registry()
    return calendar.is_market_open(symbol, dt)

def is_trading_day(symbol: str, dt: datetime | None=None) -> bool:
    """Check if given datetime is a trading day for symbol (defaults to today)."""
    if dt is None:
        dt = datetime.now(UTC)
    calendar = get_calendar_registry()
    return calendar.is_trading_day(symbol, dt)

def ensure_final_bar(symbol: str, timeframe: str) -> bool:
    """Check if final bar should be ensured for symbol/timeframe."""
    calendar = get_calendar_registry()
    return calendar.ensure_final_bar(symbol, timeframe)
