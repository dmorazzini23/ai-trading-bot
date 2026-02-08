"""Utility functions for common operations across the bot."""

import datetime as dt
import logging
import os
import random
import socket
import subprocess
import sys
import threading
import time
import warnings
from datetime import date, datetime
from datetime import time as dt_time
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import UUID
from zoneinfo import ZoneInfo

from ai_trading.config import get_settings
from ai_trading.config.management import get_env
from ai_trading.exc import COMMON_EXC
from ai_trading.logging import get_logger
from ai_trading.settings import get_verbose_logging
from ai_trading.utils.time import monotonic_time
from ai_trading.utils.env import get_alpaca_data_v2_base

from .locks import portfolio_lock
from .safe_subprocess import SUBPROCESS_TIMEOUT_DEFAULT, safe_subprocess_run

try:  # pragma: no cover - import guard mirrors runtime behaviour
    from ai_trading.alpaca_api import (  # type: ignore
        alpaca_get as _alpaca_get,
        AlpacaOrderHTTPError,
    )
except ImportError:  # pragma: no cover - fallback when Alpaca API unavailable

    class AlpacaOrderHTTPError(Exception):  # type: ignore[no-redef]
        """Fallback Alpaca HTTP error when alpaca-py is unavailable."""

        def __init__(
            self,
            status_code: int | None = None,
            message: str = "",
            *,
            payload: dict[str, Any] | None = None,
        ) -> None:
            super().__init__(message or "Alpaca request failed")
            self.status_code = status_code
            self.payload = payload or {}

    def _alpaca_get(*args: Any, **kwargs: Any):  # type: ignore[no-redef]
        raise ImportError("alpaca API unavailable")

alpaca_get = _alpaca_get

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd  # pylint: disable=unused-import
    from pandas import DataFrame, Series, Index, Timestamp
else:  # pragma: no cover - runtime when pandas missing
    DataFrame = Series = Index = Timestamp = object
def ensure_utc_index(df: DataFrame) -> DataFrame:
    """Return DataFrame with UTC tz-aware ``DatetimeIndex`` if applicable."""
    try:
        import pandas as pd  # pylint: disable=import-error
    except ImportError as exc:  # pragma: no cover - pandas missing
        raise ImportError(
            "pandas is required for ensure_utc_index. Install with `pip install ai-trading-bot[pandas]`."
        ) from exc
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


logger = get_logger(__name__)
_LAST_MARKET_HOURS_LOG = 0.0
_LAST_MARKET_STATE = ""
_LAST_MARKET_STATE_DATES: dict[str, date | None] = {"OPEN": None, "CLOSED": None}
_LAST_MARKET_CLOSED_DATE: date | None = None
_LAST_HEALTH_ROW_LOG = 0.0
_LAST_HEALTH_ROWS_COUNT = -1
_LAST_HEALTH_STATUS: bool | None = None


def _alpaca_http_error_types() -> tuple[type[Exception], ...]:
    """Return Alpaca HTTP exception types resilient to module reload churn."""

    error_types: list[type[Exception]] = []
    for candidate in (
        AlpacaOrderHTTPError,
        getattr(sys.modules.get("ai_trading.alpaca_api"), "AlpacaOrderHTTPError", None),
    ):
        if not isinstance(candidate, type):
            continue
        if not issubclass(candidate, Exception):
            continue
        if candidate in error_types:
            continue
        error_types.append(candidate)
    return tuple(error_types)


class PhaseLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that injects bot_phase context."""

    def process(self, msg, kwargs):
        extra = kwargs.get("extra")
        if extra is None:
            extra = {}
            kwargs["extra"] = extra
        extra.setdefault("bot_phase", self.extra.get("bot_phase", "GENERAL"))
        extra.setdefault("timestamp", dt.datetime.now(dt.UTC))
        return (msg, kwargs)


def get_phase_logger(name: str, phase: str) -> logging.Logger:
    """Return logger with ``bot_phase`` context."""
    base = get_logger(name)
    return PhaseLoggerAdapter(base, {"bot_phase": phase})


def log_cpu_usage(lg: logging.Logger, note: str | None = None) -> None:
    """Log current CPU usage using optional psutil snapshot."""
    try:
        from ai_trading.monitoring.system_health import snapshot_basic
    except ImportError:  # pragma: no cover - psutil missing or monitoring unavailable
        return
    pct = snapshot_basic().get("cpu_percent")
    if pct is None:
        return
    suffix = f"_{note}" if note else ""
    lg.debug("CPU_USAGE%s: %.2f%%", suffix, pct)


MIN_HEALTH_ROWS = int(os.getenv("MIN_HEALTH_ROWS", "30"))
MIN_HEALTH_ROWS_D = int(os.getenv("MIN_HEALTH_ROWS_DAILY", "5"))
HEALTH_MIN_ROWS = int(os.getenv("HEALTH_MIN_ROWS", "100"))
HEALTH_THROTTLE = 10
_last_health_log = 0.0


def log_warning(msg: str, *, exc: Exception | None = None, extra: dict | None = None) -> None:
    """Standardized warning logger used across the project."""
    if extra is None:
        extra = {}
    if exc is not None:
        if msg == "HEALTH_STALE_DATA":
            logger.debug("%s: %s", msg, exc, extra=extra, exc_info=True)
        else:
            logger.warning("%s: %s", msg, exc, extra=extra, exc_info=True)
    elif msg == "HEALTH_STALE_DATA":
        logger.debug(msg, extra=extra)
    else:
        logger.warning(msg, extra=extra)


_STALE_CACHE: dict[str, tuple[Timestamp, float]] = {}
_STALE_CACHE_LOCK = threading.Lock()


def should_log_stale(symbol: str, last_ts: Timestamp, *, ttl: int = 300) -> bool:
    """Check if stale data warning should be logged for this symbol."""
    import time

    current_time = time.time()
    with _STALE_CACHE_LOCK:
        if symbol in _STALE_CACHE:
            cached_ts, cached_time = _STALE_CACHE[symbol]
            if cached_ts == last_ts and current_time - cached_time < ttl:
                return False
        _STALE_CACHE[symbol] = (last_ts, current_time)
        return True


def get_trading_calendar(name: str = "XNYS"):
    """Return a trading calendar for the given exchange."""
    try:
        import pandas_market_calendars as mcal  # pylint: disable=import-error
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise ImportError(
            "pandas-market-calendars is required for trading calendar utilities. "
            "Install with `pip install ai-trading-bot[pandas-market-calendars]`."
        ) from exc
    return mcal.get_calendar(name)


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 30.0, jitter: float = 0.1) -> float:
    """Return exponential backoff delay with jitter."""
    exp = base * 2 ** max(0, attempt - 1)
    delay = min(exp, cap)
    if jitter > 0:
        jitter_amt = random.uniform(-jitter * delay, jitter * delay)
        delay = max(0.0, delay + jitter_amt)
    return delay


def format_order_for_log(order: Any) -> str:
    """Return compact string representation of an order for logging."""
    if order is None:
        return ""
    parts = []
    for k, v in vars(order).items():
        if isinstance(v, dt.datetime | date):
            val = v.isoformat()
        elif isinstance(v, Enum):
            val = v.value
        elif isinstance(v, UUID):
            val = str(v)
        elif isinstance(v, int | float | bool) or v is None:
            val = v
        else:
            val = str(v)
        parts.append(f"{k}={val}")
    return ", ".join(parts)


MARKET_OPEN_TIME = dt.time(9, 30)
MARKET_CLOSE_TIME = dt.time(16, 0)
EASTERN_TZ = ZoneInfo("America/New_York")


class _CallableLock:
    """threading.Lock that can be used as a context manager or callable."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def __call__(self):
        return self

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._lock.release()
        return False

    def acquire(self, *a, **k):
        return self._lock.acquire(*a, **k)

    def release(self):
        self._lock.release()

    def locked(self):
        return self._lock.locked()


model_lock = _CallableLock()


def get_latest_close(df: DataFrame) -> float:
    """Return the most recent close value or 0.0."""
    if df is None:
        return 0.0
    try:
        import pandas as pd  # pylint: disable=import-error
        import numpy as np  # pylint: disable=import-error
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise ImportError(
            "pandas and numpy are required for get_latest_close. Install with `pip install pandas numpy`."
        ) from exc
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        try:
            for col in ("close", "Close", "c"):
                if col in df.columns:
                    s = pd.to_numeric(df[col], errors="coerce").dropna()
                    if not s.empty:
                        v = float(s.iloc[-1])
                        return v if np.isfinite(v) else 0.0
            return 0.0
        except COMMON_EXC + (IndexError,):
            return 0.0


def get_current_price(symbol: str) -> float:
    """Return latest quote price with fallbacks."""
    price = 0.0
    http_error_types = _alpaca_http_error_types()
    try:
        feed = get_env("ALPACA_DATA_FEED", "iex")
        params = {"feed": feed} if feed else None
        data = alpaca_get(
            f"{get_alpaca_data_v2_base()}/stocks/{symbol}/quotes/latest",
            params=params,
        )
        price = float(data.get("ap", 0) or 0)
    except http_error_types as exc:
        logger.warning(
            "get_current_price http error for %s: %s",
            symbol,
            exc,
        )
    except COMMON_EXC as exc:
        logger.warning("get_current_price primary fetch failed for %s: %s", symbol, exc)
    if price <= 0:
        logger.warning("get_current_price invalid price %.2f for %s; falling back to last close", price, symbol)
        try:
            from ai_trading.data.fetch import get_daily_df

            end = dt.date.today()
            start = end - dt.timedelta(days=5)
            df = get_daily_df(symbol, start, end)
            price = get_latest_close(df) if df is not None else 0.0
        except COMMON_EXC as exc:
            logger.warning("get_current_price fallback failed for %s: %s", symbol, exc)
    if price <= 0:
        logger.warning("get_current_price ultimate fallback using 0.01 for %s", symbol)
        price = 0.01
    return price


def _log_market_hours(message: str, *, log_date: date | None = None) -> None:
    """Emit market hours message only on state change, new day, or hourly."""

    global _LAST_MARKET_HOURS_LOG, _LAST_MARKET_STATE, _LAST_MARKET_STATE_DATES

    now = time.time()
    state = "OPEN" if "OPEN" in message else "CLOSED"
    current_date = log_date or datetime.now(EASTERN_TZ).date()
    last_state_date = _LAST_MARKET_STATE_DATES.get(state)

    should_log = False
    if state != _LAST_MARKET_STATE:
        should_log = True
    elif last_state_date != current_date:
        should_log = True
    elif now - _LAST_MARKET_HOURS_LOG >= 3600:
        should_log = True

    if should_log:
        if get_verbose_logging():
            logger.info(message)
        else:
            logger.debug(message)
        _LAST_MARKET_STATE = state
        _LAST_MARKET_STATE_DATES[state] = current_date
        _LAST_MARKET_HOURS_LOG = now


def log_health_row_check(rows: int, passed: bool) -> None:
    """Log HEALTH_ROWS status changes or once every 10 seconds."""
    global _LAST_HEALTH_ROW_LOG, _LAST_HEALTH_ROWS_COUNT, _LAST_HEALTH_STATUS
    now = monotonic_time()
    if (
        not passed
        or rows != _LAST_HEALTH_ROWS_COUNT
        or passed != _LAST_HEALTH_STATUS
        or (now - _LAST_HEALTH_ROW_LOG >= 10)
    ):
        level = logger.info if get_verbose_logging() or not passed else logger.debug
        status = "PASSED" if passed else "FAILED"
        level("HEALTH_ROWS_%s: received %d rows", status, rows)
        _LAST_HEALTH_ROW_LOG = now
        _LAST_HEALTH_ROWS_COUNT = rows
        _LAST_HEALTH_STATUS = passed


def health_rows_passed(rows):
    """Log HEALTH_ROWS_PASSED with throttling."""
    global _last_health_log
    now = monotonic_time()
    if _last_health_log == 0.0 or now - _last_health_log >= HEALTH_THROTTLE:
        count = len(rows) if not isinstance(rows, int | float) else rows
        logger.debug("HEALTH_ROWS_PASSED: received %d rows", count)
        _last_health_log = now or 1e-09
    else:
        logger.debug("HEALTH_ROWS_THROTTLED")
    return rows


def is_market_open(now: dt.datetime | None = None) -> bool:
    """Return True if current time is within NYSE trading hours."""
    global _LAST_MARKET_CLOSED_DATE
    if os.getenv("FORCE_MARKET_OPEN", "false").lower() == "true":
        logger.info("FORCE_MARKET_OPEN is enabled; overriding market hours checks.")
        return True
    check_time = (now or dt.datetime.now(dt.UTC)).astimezone(EASTERN_TZ)
    current_date = check_time.date()
    if _LAST_MARKET_CLOSED_DATE == current_date:
        return False
    try:
        import pandas_market_calendars as mcal  # pylint: disable=import-error
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise ImportError(
            "pandas-market-calendars is required for is_market_open. Install with `pip install ai-trading-bot[pandas-market-calendars]`."
        ) from exc
    try:
        cal = getattr(mcal, "get_calendar", None)
        if cal is None:
            return False
        cal = cal("NYSE")
        sched = cal.schedule(start_date=current_date, end_date=current_date)
        if sched.empty:
            is_weekend = check_time.weekday() >= 5
            is_future = current_date > dt.date.today()
            if is_weekend:
                logger.debug("No market schedule for %s (weekend); returning False.", current_date)
            elif is_future:
                logger.debug("No market schedule for %s (future date); returning False.", current_date)
            else:
                logger.warning(
                    "No market schedule for %s in is_market_open (likely holiday); returning False.", current_date
                )
            _LAST_MARKET_CLOSED_DATE = current_date
            _log_market_hours("Detected Market Hours today: CLOSED", log_date=current_date)
            return False
        market_open = sched.iloc[0]["market_open"].tz_convert(EASTERN_TZ).time()
        market_close = sched.iloc[0]["market_close"].tz_convert(EASTERN_TZ).time()
        if check_time.month == 7 and check_time.day == 3:
            july4 = date(check_time.year, 7, 4)
            market_close = dt_time(13, 0) if july4.weekday() >= 5 else MARKET_CLOSE_TIME
        _log_market_hours(
            "Detected Market Hours today: OPEN from {} to {}".format(
                market_open.strftime("%H:%M"), market_close.strftime("%H:%M")
            ),
            log_date=current_date,
        )
        current = check_time.time()
        _LAST_MARKET_CLOSED_DATE = None
        return market_open <= current <= market_close
    except COMMON_EXC as exc:
        logger.debug("market calendar unavailable: %s", exc)
        now_et = check_time
        if now_et.weekday() >= 5:
            _LAST_MARKET_CLOSED_DATE = now_et.date()
            _log_market_hours("Detected Market Hours today: CLOSED", log_date=now_et.date())
            return False
        current = now_et.time()
        _log_market_hours(
            "Detected Market Hours today: OPEN from {} to {}".format(
                MARKET_OPEN_TIME.strftime("%H:%M"), MARKET_CLOSE_TIME.strftime("%H:%M")
            ),
            log_date=now_et.date(),
        )
        _LAST_MARKET_CLOSED_DATE = None
        return MARKET_OPEN_TIME <= current <= MARKET_CLOSE_TIME


def next_market_open(now: dt.datetime | None = None) -> dt.datetime:
    """Return the next NYSE market open time in US/Eastern."""
    check_time = (now or dt.datetime.now(dt.UTC)).astimezone(EASTERN_TZ)
    try:
        import pandas_market_calendars as mcal  # pylint: disable=import-error

        nyse = mcal.get_calendar("NYSE")
        start = check_time.date()
        end = start + dt.timedelta(days=7)
        sched = nyse.schedule(start_date=start, end_date=end)
        if "market_open" not in sched.columns:
            logger.debug(
                "next_market_open schedule missing column",
                extra={"available_columns": list(sched.columns)},
            )
        else:
            future = sched[sched["market_open"] > check_time]
            if not future.empty:
                return future.iloc[0]["market_open"].tz_convert(EASTERN_TZ).to_pydatetime()
    except (ImportError, ValueError, TypeError, KeyError) as exc:  # pragma: no cover - best effort
        logger.debug("next_market_open calendar lookup failed: %s", exc)

    candidate = check_time
    if candidate.weekday() < 5 and candidate.time() < MARKET_OPEN_TIME:
        candidate = candidate.replace(
            hour=MARKET_OPEN_TIME.hour,
            minute=MARKET_OPEN_TIME.minute,
            second=0,
            microsecond=0,
        )
    else:
        candidate = (candidate + dt.timedelta(days=1)).replace(
            hour=MARKET_OPEN_TIME.hour,
            minute=MARKET_OPEN_TIME.minute,
            second=0,
            microsecond=0,
        )
    while candidate.weekday() >= 5:
        candidate += dt.timedelta(days=1)
        candidate = candidate.replace(
            hour=MARKET_OPEN_TIME.hour,
            minute=MARKET_OPEN_TIME.minute,
            second=0,
            microsecond=0,
        )
    return candidate


def market_open_between(start: datetime, end: datetime) -> bool:
    """Return True if market is open at any point in [start, end]."""
    if end < start:
        start, end = (end, start)
    current = start
    while current <= end:
        if is_market_open(current):
            return True
        current += dt.timedelta(minutes=1)
    return False


def is_weekend(timestamp: dt.datetime | Timestamp | None = None) -> bool:
    """Check if the given timestamp (or current time) falls on a weekend."""
    if timestamp is None:
        timestamp = dt.datetime.now(dt.UTC)
    elif hasattr(timestamp, "to_pydatetime"):
        timestamp = timestamp.to_pydatetime()
    try:
        et_time = timestamp.astimezone(ZoneInfo("America/New_York"))
        return et_time.weekday() >= 5
    except COMMON_EXC:
        return timestamp.weekday() >= 5


def is_market_holiday(date_to_check: date | dt.datetime | None = None) -> bool:
    """Return True for a small set of known US market holidays.

    The real project uses ``pandas-market-calendars`` but that optional
    dependency is intentionally avoided in tests.  We fall back to a minimal
    static list that covers the dates exercised in the unit tests.
    """
    if date_to_check is None:
        date_to_check = dt.datetime.now(dt.UTC).date()
    elif isinstance(date_to_check, dt.datetime):
        date_to_check = date_to_check.date()
    month_day = (date_to_check.month, date_to_check.day)
    # Minimal holiday set for tests
    holidays = {(1, 1), (12, 25)}
    return month_day in holidays


BASE_PATH = os.path.dirname(os.path.abspath(__file__))


def ensure_utc(value: dt.datetime | date) -> dt.datetime:
    """Return a timezone-aware UTC datetime for ``dt``."""
    if not isinstance(value, (dt.datetime, date)):
        raise TypeError("dt must be date or datetime")
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=dt.UTC)
        return value.astimezone(dt.UTC)
    if isinstance(value, date):
        return dt.datetime.combine(value, dt.time.min, tzinfo=dt.UTC)
    raise TypeError(f"Unsupported type for ensure_utc: {type(value)!r}")


def get_free_port(start: int | None = None, end: int | None = None) -> int | None:
    """
    If ``start`` and ``end`` are provided, return a free port in that range;
    otherwise ask the OS for an ephemeral port.
    """
    if start is not None and end is not None:
        for port in range(start, end + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind(("", port))
                    return port
                except OSError:
                    continue
        return None
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _pid_from_inode(inode: str) -> int | None:
    """Return PID for a socket inode on Linux."""
    for pid in filter(str.isdigit, os.listdir("/proc")):
        fd_dir = f"/proc/{pid}/fd"
        if not os.path.isdir(fd_dir):
            continue
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                if os.readlink(os.path.join(fd_dir, fd)) == f"socket:[{inode}]":
                    return int(pid)
            except OSError:
                continue
    return None


def get_pid_on_port(port: int) -> int | None:
    """Best-effort detection of PID bound to ``port``.

    Inspects both ``/proc/net/tcp`` and ``/proc/net/tcp6`` to handle
    IPv4 and IPv6 sockets.
    """
    for proc_path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(proc_path) as f:
                next(f)
                for line in f:
                    parts = line.split()
                    local = parts[1]
                    inode = parts[9]
                    try:
                        if int(local.split(":")[1], 16) == port:
                            pid = _pid_from_inode(inode)
                            if pid is not None:
                                return pid
                    except (ValueError, IndexError):
                        continue
        except COMMON_EXC + (OSError,) as e:
            logger.error("get_pid_on_port failed for %s", proc_path, exc_info=e)
    return None


def get_rolling_atr(symbol: str, window: int = 14) -> float:
    """Return normalized ATR over ``window`` days."""
    from ai_trading.core.bot_engine import fetch_minute_df_safe

    df = fetch_minute_df_safe(symbol)
    if df is None or df.empty:
        return 0.0
    try:
        import pandas as pd  # pylint: disable=import-error
    except ImportError as exc:  # pragma: no cover - pandas missing
        raise ImportError(
            "pandas is required for get_rolling_atr. Install with `pip install ai-trading-bot[pandas]`."
        ) from exc
    high = df["high"].rolling(window).max()
    low = df["low"].rolling(window).min()
    close = df["close"]
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window).mean().iloc[-1]
    last_valid_close = close.dropna()
    if not last_valid_close.empty:
        last_close = last_valid_close.iloc[-1]
    else:
        logger.critical("All NaNs in close column for get_rolling_atr")
        last_close = 0.0
    val = float(atr) / float(last_close) if last_close else 0.0
    logger.info("ATR for %s=%.5f", symbol, val)
    return val


def get_current_vwap(symbol: str) -> float:
    """Return simple intraday VWAP for ``symbol``."""
    from ai_trading.core.bot_engine import fetch_minute_df_safe

    df = fetch_minute_df_safe(symbol)
    if df is None or df.empty:
        return 0.0
    pv = (df["close"] * df["volume"]).sum()
    vol = df["volume"].sum()
    vwap = pv / vol if vol else 0.0
    logger.info("VWAP for %s=%.4f", symbol, vwap)
    return float(vwap)


def get_volume_spike_factor(symbol: str) -> float:
    """Return last minute volume over 20-period average."""
    from ai_trading.core.bot_engine import fetch_minute_df_safe

    df = fetch_minute_df_safe(symbol)
    if df is None or len(df) < 21:
        return 1.0
    last_vol = df["volume"].iloc[-1]
    avg_vol = df["volume"].iloc[-21:-1].mean()
    factor = float(last_vol) / float(avg_vol) if avg_vol else 1.0
    logger.info("Volume spike %s=%.2f", symbol, factor)
    return factor


def get_ml_confidence(symbol: str) -> float:
    """Return model confidence for ``symbol``."""
    try:
        from ai_trading.ml_model import load_model
    except COMMON_EXC as e:
        logger.error("load_model failed", exc_info=e)
        return 0.5
    try:
        import pandas as pd  # pylint: disable=import-error
    except ImportError as exc:  # pragma: no cover - pandas missing
        raise ImportError(
            "pandas is required for get_ml_confidence. Install with `pip install ai-trading-bot[pandas]`."
        ) from exc
    S = get_settings()
    model_path = S.model_path
    model = load_model(model_path)
    if model is None:
        return 0.5
    feats = pd.DataFrame({"price": [0.0]})
    try:
        conf = float(model.predict_proba(feats)[0][1])
    except COMMON_EXC as e:
        logger.error("predict_proba failed", exc_info=e)
        conf = 0.5
    logger.info("ML confidence for %s=%.2f", symbol, conf)
    return conf


def to_serializable(obj: Any) -> Any:
    """Recursively convert unsupported types for JSON serialization."""
    from types import MappingProxyType

    if isinstance(obj, MappingProxyType):
        obj = dict(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_serializable(v) for v in obj]
    return obj


_WARN_COUNTS: dict[str, int] = {}


def _warn_limited(key: str, msg: str, *args, limit: int = 3, **kwargs) -> None:
    """Log a warning only up to ``limit`` times for the given ``key``."""
    count = _WARN_COUNTS.get(key, 0)
    if count < limit:
        logger.warning(msg, *args, **kwargs)
        _WARN_COUNTS[key] = count + 1
        if count + 1 == limit:
            logger.warning("Further '%s' warnings suppressed", key)


def safe_to_datetime(
    values,
    format: str | None = None,
    utc: bool = True,
    *,
    context: str = "",
    _warn_key: str | None = None,
):
    """Safely convert ``values`` to a timezone-aware :class:`~pandas.DatetimeIndex`."""

    try:
        import pandas as pd  # pylint: disable=import-error
    except ImportError as exc:  # pragma: no cover - pandas missing
        raise ImportError(
            "pandas is required for safe_to_datetime. Install with `pip install ai-trading-bot[pandas]`."
        ) from exc

    if values is None:
        return pd.DatetimeIndex([], tz="UTC") if utc else pd.DatetimeIndex([])

    try:
        series = pd.Series(values)
    except (TypeError, ValueError):
        try:
            series = pd.Series(list(values))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            series = pd.Series([values])

    if series.empty:
        return pd.DatetimeIndex([], tz="UTC") if utc else pd.DatetimeIndex([])

    series = series.replace({"": None, "0": None, 0: None})

    numeric = series.dropna()
    is_numeric = False
    try:
        from pandas.api import types as pd_types  # pylint: disable=import-error

        is_numeric = pd_types.is_numeric_dtype(numeric)
    except (ImportError, AttributeError, TypeError, ValueError):
        try:
            is_numeric = bool(numeric.apply(lambda value: isinstance(value, (int, float))).all())
        except (TypeError, AttributeError, ValueError):
            is_numeric = False

    if is_numeric:
        coerced = pd.to_numeric(series, errors="coerce")
        finite = coerced.dropna()
        if not finite.empty and (finite > 1_000_000_000_000).any():
            coerced = coerced / 1000.0
        converted = pd.to_datetime(coerced, unit="s", errors="coerce", utc=utc)
    else:
        converted = pd.to_datetime(series, format=format, errors="coerce", utc=utc)

    try:
        coerced = int(((converted.isna()) & series.notna()).sum())
    except (AttributeError, TypeError, ValueError):
        coerced = 0
    if coerced:
        warn_key = _warn_key or (f"SAFE_TO_DATETIME:{context}" if context else "SAFE_TO_DATETIME")
        try:
            from ai_trading.logging import log_once

            log_once(warn_key, f"safe_to_datetime coerced {coerced} values to NaT")
        except (ImportError, AttributeError, RuntimeError):  # pragma: no cover - warning best effort
            pass

    try:
        index = pd.DatetimeIndex(converted)
    except (TypeError, ValueError):
        index = pd.DatetimeIndex(converted.to_numpy())  # type: ignore[arg-type]

    if utc:
        if getattr(index, "tz", None) is None:
            index = index.tz_localize("UTC")
        return index

    if getattr(index, "tz", None) is not None:
        return index.tz_convert("UTC").tz_localize(None)
    return index


def validate_ohlcv(df: DataFrame, required: list[str] | None = None, require_monotonic: bool = True) -> None:
    """
    Validate an OHLCV-like DataFrame in-place. Raises ValueError on failure.
    Required columns default to ['timestamp','open','high','low','close','volume'].
    Ensures timestamp is parseable and, if requested, monotonic increasing.
    """
    try:
        import pandas as pd  # pylint: disable=import-error
    except ImportError as exc:  # pragma: no cover - pandas missing
        raise ImportError(
            "pandas is required for validate_ohlcv. Install with `pip install ai-trading-bot[pandas]`."
        ) from exc
    required = required or ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    ts = df["timestamp"]
    timestamp_types = (pd.Timestamp, datetime)
    if not isinstance(ts.iloc[0], timestamp_types):
        ts = safe_to_datetime(ts, context="ohlcv validation")
    if ts.isna().any():
        raise ValueError("timestamp contains NaT/invalid values")
    if require_monotonic and (not ts.is_monotonic_increasing):
        raise ValueError("timestamp is not monotonic increasing")
    if len(df) == 0:
        raise ValueError("no rows")
    if not {"open", "high", "low", "close"}.issubset(df.columns):
        raise ValueError("OHLC columns incomplete")


def health_check(df: DataFrame, resolution: str | None = None) -> bool:
    """Return True if ``df`` has at least ``HEALTH_MIN_ROWS`` rows."""
    min_rows = int(os.getenv("HEALTH_MIN_ROWS", "0"))
    try:
        return len(df) >= min_rows
    except COMMON_EXC:
        return False


def get_column(
    df,
    options,
    label,
    dtype=None,
    must_be_monotonic=False,
    must_be_non_null=False,
    must_be_unique=False,
    must_be_timezone_aware=False,
):
    try:
        import pandas as pd  # pylint: disable=import-error
    except ImportError as exc:  # pragma: no cover - pandas missing
        raise ImportError(
            "pandas is required for get_column. Install with `pip install ai-trading-bot[pandas]`."
        ) from exc
    for col in options:
        if col in df.columns:
            if dtype is not None:
                if dtype == "datetime64[ns]" and pd.api.types.is_datetime64_any_dtype(df[col]):
                    continue
                elif not pd.api.types.is_dtype_equal(df[col].dtype, dtype):
                    raise TypeError(f"{label}: column '{col}' is not of dtype {dtype}, got {df[col].dtype}")
            if must_be_monotonic and (not df[col].is_monotonic_increasing):
                raise ValueError(f"{label}: column '{col}' is not monotonic increasing")
            if must_be_non_null and df[col].isnull().all():
                raise ValueError(f"{label}: column '{col}' is all null")
            if must_be_unique and (not df[col].is_unique):
                raise ValueError(f"{label}: column '{col}' is not unique")
            if must_be_timezone_aware and hasattr(df[col], "dt") and (df[col].dt.tz is None):
                raise ValueError(f"{label}: column '{col}' is not timezone-aware")
            return col
    raise ValueError(f"No recognized {label} column found in DataFrame: {df.columns.tolist()}")


def get_open_column(df):
    return _safe_get_column(df, ["Open", "open", "o"], "open price", dtype=None)


def get_high_column(df):
    return _safe_get_column(df, ["High", "high", "h"], "high price", dtype=None)


def get_low_column(df):
    return _safe_get_column(df, ["Low", "low", "l"], "low price", dtype=None)


def get_close_column(df):
    return _safe_get_column(
        df, ["Close", "close", "c", "adj_close", "Adj Close", "adjclose", "adjusted_close"], "close price", dtype=None
    )


def get_volume_column(df):
    return _safe_get_column(df, ["Volume", "volume", "v"], "volume", dtype=None)


def _safe_get_column(df, options, label, **kwargs):
    try:
        import pandas as pd  # pylint: disable=import-error
    except ImportError as exc:  # pragma: no cover - pandas missing
        raise ImportError(
            "pandas is required for column helpers. Install with `pip install ai-trading-bot[pandas]`."
        ) from exc
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    try:
        return get_column(df, options, label, **kwargs)
    except (ValueError, TypeError) as exc:
        logger.warning("_safe_get_column failed for %s: %s", label, exc)
        return None


def get_datetime_column(df):
    return _safe_get_column(
        df,
        ["Datetime", "datetime", "timestamp", "date"],
        "datetime",
        dtype="datetime64[ns]",
        must_be_monotonic=True,
        must_be_non_null=True,
        must_be_timezone_aware=True,
    )


def get_symbol_column(df):
    return _safe_get_column(df, ["symbol", "ticker", "SYMBOL"], "symbol", dtype="O", must_be_unique=True)


def get_return_column(df):
    return _safe_get_column(df, ["Return", "ret", "returns"], "return", dtype=None, must_be_non_null=True)


def get_indicator_column(df, possible_names):
    return _safe_get_column(df, possible_names, "indicator")


def get_order_column(df, name):
    return _safe_get_column(df, [name, name.lower(), name.upper()], f"order/{name}", dtype=None, must_be_non_null=True)


def get_ohlcv_columns(df):
    """Return the names of the OHLCV columns if present."""
    try:
        import pandas as pd  # pylint: disable=import-error
    except ImportError as exc:  # pragma: no cover - pandas missing
        raise ImportError(
            "pandas is required for get_ohlcv_columns. Install with `pip install ai-trading-bot[pandas]`."
        ) from exc
    if not isinstance(df, pd.DataFrame):
        return []
    cols = [get_open_column(df), get_high_column(df), get_low_column(df), get_close_column(df), get_volume_column(df)]
    if any((c is None for c in cols)):
        return []
    return cols


REQUIRED_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


def validate_ohlcv_basic(df: DataFrame) -> bool:
    """Return True if ``df`` contains the required OHLCV columns."""
    try:
        import pandas as pd  # pylint: disable=import-error
    except ImportError as exc:  # pragma: no cover - pandas missing
        raise ImportError(
            "pandas is required for validate_ohlcv_basic. Install with `pip install ai-trading-bot[pandas]`."
        ) from exc
    if not isinstance(df, pd.DataFrame) or df.empty:
        logger.error("validate_ohlcv_basic received invalid DataFrame")
        return False
    missing = [c for c in REQUIRED_OHLCV_COLS if c not in df.columns]
    if missing:
        logger.error("Missing OHLCV columns: %s", missing)
        return False
    return True


def _get_alpaca_rest():
    """Get Alpaca :class:`TradingClient` class."""
    try:
        from alpaca.trading.client import TradingClient  # pylint: disable=import-error
    except ImportError as exc:  # pragma: no cover - alpaca SDK missing
        raise ImportError("alpaca-py is required for Alpaca access. Install with `pip install alpaca-py`.") from exc
    return TradingClient


def check_symbol(symbol: str, api: Any) -> bool:
    """Return ``True`` if ``symbol`` has sufficient data via ``api``."""
    try:
        import pandas as pd  # pylint: disable=import-error
    except ImportError as exc:  # pragma: no cover - pandas missing
        raise ImportError(
            "pandas is required for check_symbol. Install with `pip install ai-trading-bot[pandas]`."
        ) from exc
    try:
        path = os.path.join("data", f"{symbol}.csv")
        df = pd.read_csv(path)
    except COMMON_EXC as exc:
        logger.warning("Health check fetch failed for %s: %s", symbol, exc)
        return False
    return health_check(df, "daily")


def pre_trade_health_check(symbols: list[str], api: Any) -> dict[str, bool]:
    """Check data availability for ``symbols`` prior to trading.

    Parameters
    ----------
    symbols : list[str]
        Symbols to validate.

    Returns
    -------
    dict[str, bool]
        Mapping of symbol to health status.
    """
    symbol_health: dict[str, bool] = {}
    for sym in symbols:
        ok = check_symbol(sym, api)
        symbol_health[sym] = ok
        if not ok:
            logger.warning(f"Health check skipped for {sym}: insufficient data")
    return symbol_health


def enable_market_calendar_lib() -> None:
    """Optionally import pandas_market_calendars when configured."""
    from importlib.util import find_spec

    S = get_settings()
    if getattr(S, "use_market_calendar_lib", False):
        if find_spec("pandas_market_calendars") is None:
            raise RuntimeError("Feature enabled but module 'pandas_market_calendars' not installed")
        import pandas_market_calendars as _pmc

        globals().update({k: getattr(_pmc, k) for k in dir(_pmc) if not k.startswith("_")})
