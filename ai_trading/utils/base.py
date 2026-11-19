"""Utility functions for common operations across the bot."""

import datetime as dt
import logging
import os
import random
import socket
import subprocess
import threading
import time
import warnings
from datetime import date, datetime
from datetime import time as dt_time
from typing import TYPE_CHECKING, Any, cast
from zoneinfo import ZoneInfo

from ai_trading.config.management import get_env
from ai_trading.exc import COMMON_EXC
from ai_trading.logging import get_logger
from ai_trading.settings import get_verbose_logging
from ai_trading.utils.time import monotonic_time

from .locks import portfolio_lock
from .safe_subprocess import SUBPROCESS_TIMEOUT_DEFAULT, safe_subprocess_run

from ai_trading.alpaca_api import (  # type: ignore
    alpaca_get as _alpaca_get,
)

alpaca_get = _alpaca_get

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd  # pylint: disable=unused-import
    from pandas import DataFrame, Series, Index, Timestamp
else:  # pragma: no cover - runtime when pandas missing
    DataFrame = Series = Index = Timestamp = object


logger = get_logger(__name__)
_LAST_MARKET_HOURS_LOG = 0.0
_LAST_MARKET_STATE = ""
_LAST_MARKET_STATE_DATES: dict[str, date | None] = {"OPEN": None, "CLOSED": None}
_LAST_MARKET_CLOSED_DATE: date | None = None
_LAST_HEALTH_ROW_LOG = 0.0
_LAST_HEALTH_ROWS_COUNT = -1
_LAST_HEALTH_STATUS: bool | None = None


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


def get_phase_logger(name: str, phase: str) -> PhaseLoggerAdapter:
    """Return logger with ``bot_phase`` context."""
    base = get_logger(name)
    return PhaseLoggerAdapter(base, {"bot_phase": phase})


MIN_HEALTH_ROWS = int(get_env("MIN_HEALTH_ROWS", "30", cast=str, resolve_aliases=False))
MIN_HEALTH_ROWS_D = int(
    get_env("MIN_HEALTH_ROWS_DAILY", "5", cast=str, resolve_aliases=False)
)
HEALTH_MIN_ROWS = int(get_env("HEALTH_MIN_ROWS", "100", cast=str, resolve_aliases=False))
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


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 30.0, jitter: float = 0.1) -> float:
    """Return exponential backoff delay with jitter."""
    exp = base * 2 ** max(0, attempt - 1)
    delay = min(exp, cap)
    if jitter > 0:
        jitter_amt = random.uniform(-jitter * delay, jitter * delay)
        delay = max(0.0, delay + jitter_amt)
    return float(delay)


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

    def __exit__(self, exc_type, exc, _tb):
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
    if str(
        get_env("FORCE_MARKET_OPEN", "false", cast=str, resolve_aliases=False)
    ).lower() == "true":
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
        _log_market_hours(
            "Detected Market Hours today: OPEN from {} to {}".format(
                market_open.strftime("%H:%M"), market_close.strftime("%H:%M")
            ),
            log_date=current_date,
        )
        current = check_time.time()
        _LAST_MARKET_CLOSED_DATE = None
        return bool(market_open <= current <= market_close)
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
        return bool(MARKET_OPEN_TIME <= current <= MARKET_CLOSE_TIME)


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
                return cast(
                    datetime,
                    future.iloc[0]["market_open"].tz_convert(EASTERN_TZ).to_pydatetime(),
                )
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
        return int(sock.getsockname()[1])


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
            from ai_trading.logging import logger_once

            logger_once.warning(
                f"safe_to_datetime coerced {coerced} values to NaT",
                key=warn_key,
            )
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


def health_check(df: DataFrame, resolution: str | None = None) -> bool:
    """Return True if ``df`` has at least ``HEALTH_MIN_ROWS`` rows."""
    min_rows = int(get_env("HEALTH_MIN_ROWS", "0", cast=str, resolve_aliases=False))
    try:
        return len(df) >= min_rows
    except COMMON_EXC:
        return False


REQUIRED_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


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

