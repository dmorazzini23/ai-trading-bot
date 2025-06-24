"""Utility functions for common operations across the bot."""

import logging
import os
import socket
import warnings
import datetime as dt
from datetime import date, time, timezone

import pandas as pd
from typing import Any
from zoneinfo import ZoneInfo
import threading

try:
    from tzlocal import get_localzone
except ImportError:  # pragma: no cover - optional dependency
    logging.warning("tzlocal not installed; defaulting to UTC")

    def get_localzone() -> ZoneInfo:
        return ZoneInfo("UTC")


logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)

MIN_HEALTH_ROWS = int(os.getenv("MIN_HEALTH_ROWS", "30"))
MIN_HEALTH_ROWS_D = int(os.getenv("MIN_HEALTH_ROWS_DAILY", "5"))


def log_warning(
    msg: str, *, exc: Exception | None = None, extra: dict | None = None
) -> None:
    """Standardized warning logger used across the project."""
    if extra is None:
        extra = {}
    if exc is not None:
        logger.warning("%s: %s", msg, exc, extra=extra, exc_info=True)
    else:
        logger.warning(msg, extra=extra)


MARKET_OPEN_TIME = time(9, 30)
MARKET_CLOSE_TIME = time(16, 0)
EASTERN_TZ = ZoneInfo("America/New_York")

# Lock protecting portfolio state across threads
portfolio_lock = threading.Lock()
# Lock protecting model updates


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


def get_latest_close(df: pd.DataFrame) -> float:
    """Return last closing price or 1.0 if unavailable."""
    if df is None or df.empty:
        return 1.0
    if "close" in df.columns:
        last = df["close"].iloc[-1]
    else:
        return 1.0
    if pd.isna(last) or last == 0:
        return 1.0
    return float(last)


def is_market_open(now: dt.datetime | None = None) -> bool:
    """Return True if current time is within NYSE trading hours."""
    try:
        import pandas_market_calendars as mcal

        check_time = (
            now or dt.datetime.now(tz=EASTERN_TZ)
        ).astimezone(EASTERN_TZ)
        cal = getattr(mcal, "get_calendar", None)
        if cal is None:
            raise AttributeError
        cal = cal("NYSE")
        sched = cal.schedule(
            start_date=check_time.date(),
            end_date=check_time.date(),
        )
        if sched.empty:
            logger.warning(
                "No market schedule for %s in is_market_open; returning False.",
                check_time.date(),
            )
            return False  # holiday or weekend
        market_open = sched.iloc[0]["market_open"].tz_convert(EASTERN_TZ).time()
        market_close = sched.iloc[0]["market_close"].tz_convert(EASTERN_TZ).time()
        current = check_time.time()
        return market_open <= current <= market_close
    except (ImportError, AttributeError, KeyError, ValueError) as exc:
        logger.debug("market calendar unavailable: %s", exc)
        # Fallback to simple weekday/time check when calendar unavailable
        now_et = (
            now or dt.datetime.now(tz=EASTERN_TZ)
        ).astimezone(EASTERN_TZ)
        if now_et.weekday() >= 5:
            return False
        current = now_et.time()
        return MARKET_OPEN_TIME <= current <= MARKET_CLOSE_TIME


BASE_PATH = os.path.dirname(os.path.abspath(__file__))


def ensure_utc(value: dt.datetime | date) -> dt.datetime:
    """Return a timezone-aware UTC datetime for ``dt``."""
    assert isinstance(value, (dt.datetime, date)), "dt must be date or datetime"
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, date):
        return dt.datetime.combine(value, time.min, tzinfo=timezone.utc)
    raise TypeError(f"Unsupported type for ensure_utc: {type(value)!r}")


def get_free_port(start: int = 9200, end: int = 9300) -> int | None:
    """Return an available TCP port in the range [start, end]."""
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("", port))
                return port
            except OSError:
                continue
    return None


def to_serializable(obj: Any) -> Any:
    """Recursively convert unsupported types for JSON serialization."""
    from types import MappingProxyType

    if isinstance(obj, MappingProxyType):
        obj = dict(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
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


def safe_to_datetime(arr, format="%Y-%m-%d %H:%M:%S", utc=True, *, context: str = ""):
    """Safely convert an iterable of date strings to ``DatetimeIndex``."""

    if arr is None:
        return pd.DatetimeIndex([], tz="UTC")

    # if someone passed (symbol, Timestamp), extract the Timestamp
    if isinstance(arr, tuple) and len(arr) == 2 and isinstance(arr[1], pd.Timestamp):
        arr = arr[1]
    elif (
        isinstance(arr, (list, pd.Index, pd.Series))
        and len(arr) > 0
        and isinstance(arr[0], tuple)
    ):
        arr = [x[1] if isinstance(x, tuple) and len(x) == 2 else x for x in arr]

    try:
        return pd.to_datetime(arr, format=format, utc=utc)
    except (TypeError, ValueError) as exc:
        logger.warning("safe_to_datetime coercing invalid values – %s", exc)
        try:
            return pd.to_datetime(arr, format=format, errors="coerce", utc=utc)
        except (TypeError, ValueError) as exc2:
            logger.error("safe_to_datetime failed: %s", exc2)
            return pd.DatetimeIndex([pd.NaT] * len(arr), tz="UTC")


def health_check(df: pd.DataFrame, resolution: str) -> bool:
    rows = len(df)
    if resolution == "minute":
        if rows < MIN_HEALTH_ROWS:
            logger.warning(
                f"HEALTH_INSUFFICIENT_ROWS: got {rows}, need {MIN_HEALTH_ROWS}"
            )
            return False
    else:
        if rows < MIN_HEALTH_ROWS_D:
            logger.warning(
                f"DAILY_HEALTH_INSUFFICIENT_ROWS: got {rows}, need {MIN_HEALTH_ROWS_D}"
            )
            return False
# End of health_check
    return True


# Generic robust column getter with validation


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
    for col in options:
        if col in df.columns:
            if dtype is not None:
                if dtype == "datetime64[ns]" and pd.api.types.is_datetime64_any_dtype(
                    df[col]
                ):
                    ...
                elif not pd.api.types.is_dtype_equal(df[col].dtype, dtype):
                    raise TypeError(
                        f"{label}: column '{col}' is not of dtype {dtype}, got {df[col].dtype}"
                    )
            if must_be_monotonic and not df[col].is_monotonic_increasing:
                raise ValueError(f"{label}: column '{col}' is not monotonic increasing")
            if must_be_non_null and df[col].isnull().all():
                raise ValueError(f"{label}: column '{col}' is all null")
            if must_be_unique and not df[col].is_unique:
                raise ValueError(f"{label}: column '{col}' is not unique")
            if (
                must_be_timezone_aware
                and hasattr(df[col], "dt")
                and df[col].dt.tz is None
            ):
                raise ValueError(f"{label}: column '{col}' is not timezone-aware")
            return col
    raise ValueError(
        f"No recognized {label} column found in DataFrame: {df.columns.tolist()}"
    )


# OHLCV helpers


def get_open_column(df):
    return get_column(df, ["Open", "open", "o"], "open price", dtype=None)


def get_high_column(df):
    return get_column(df, ["High", "high", "h"], "high price", dtype=None)


def get_low_column(df):
    return get_column(df, ["Low", "low", "l"], "low price", dtype=None)


def get_close_column(df):
    return get_column(
        df,
        ["Close", "close", "c", "adj_close", "Adj Close", "adjclose", "adjusted_close"],
        "close price",
        dtype=None,
    )


def get_volume_column(df):
    return get_column(df, ["Volume", "volume", "v"], "volume", dtype=None)


# Datetime helper with advanced checks


def _safe_get_column(df, options, label, **kwargs):
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


# Ticker/symbol column


def get_symbol_column(df):
    return _safe_get_column(
        df, ["symbol", "ticker", "SYMBOL"], "symbol", dtype="O", must_be_unique=True
    )


# Return/returns column


def get_return_column(df):
    return _safe_get_column(
        df, ["Return", "ret", "returns"], "return", dtype=None, must_be_non_null=True
    )


# Indicator column (pass a list, e.g. ["SMA", "sma", "EMA", ...])


def get_indicator_column(df, possible_names):
    return _safe_get_column(df, possible_names, "indicator")


# Order/trade columns


def get_order_column(df, name):
    return _safe_get_column(
        df,
        [name, name.lower(), name.upper()],
        f"order/{name}",
        dtype=None,
        must_be_non_null=True,
    )


def get_ohlcv_columns(df):
    """Return the names of the OHLCV columns if present."""

    if not isinstance(df, pd.DataFrame):
        return []
    try:
        return [
            get_open_column(df),
            get_high_column(df),
            get_low_column(df),
            get_close_column(df),
            get_volume_column(df),
        ]
    except KeyError:
        return []


from typing import List, Dict
try:
    from alpaca_trade_api.rest import REST
except Exception:  # pragma: no cover - optional dependency
    REST = object  # type: ignore


def check_symbol(symbol: str, api: REST) -> bool:
    """Return ``True`` if ``symbol`` has sufficient data via ``api``."""
    try:
        path = os.path.join("data", f"{symbol}.csv")
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - I/O error
        logging.warning("Health check fetch failed for %s: %s", symbol, exc)
        return False
    return health_check(df, "daily")


def pre_trade_health_check(symbols: List[str], api: REST) -> Dict[str, bool]:
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

    symbol_health: Dict[str, bool] = {}
    for sym in symbols:
        ok = check_symbol(sym, api)
        symbol_health[sym] = ok
        if not ok:
            logging.warning(
                f"Health check skipped for {sym}: insufficient data"
            )
    return symbol_health
