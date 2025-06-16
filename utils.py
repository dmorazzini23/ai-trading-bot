"""Utility functions for common operations across the bot."""

import warnings
import os
import re
import logging

import pandas as pd
import numpy as np
from datetime import datetime, date, time, timezone
from typing import Iterable, Any

try:
    from tzlocal import get_localzone
except ImportError:  # pragma: no cover - optional dependency
    import pytz

    logging.warning("tzlocal not installed; defaulting to UTC")

    def get_localzone():
        return pytz.UTC

logger = logging.getLogger(__name__)


from zoneinfo import ZoneInfo
import threading

warnings.filterwarnings("ignore", category=FutureWarning)

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


def is_market_open(now: datetime | None = None) -> bool:
    """Return True if current time is within NYSE trading hours."""
    try:
        import pandas_market_calendars as mcal

        check_time = (now or datetime.now(tz=EASTERN_TZ)).astimezone(EASTERN_TZ)
        cal = getattr(mcal, "get_calendar", None)
        if cal is None:
            raise AttributeError
        cal = cal("NYSE")
        sched = cal.schedule(
            start_date=check_time.date(), end_date=check_time.date()
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
    except Exception as e:
        logger.debug("market calendar unavailable: %s", e)
        # Fallback to simple weekday/time check when calendar unavailable
        now_et = (now or datetime.now(tz=EASTERN_TZ)).astimezone(EASTERN_TZ)
        if now_et.weekday() >= 5:
            return False
        current = now_et.time()
        return MARKET_OPEN_TIME <= current <= MARKET_CLOSE_TIME


BASE_PATH = os.path.dirname(os.path.abspath(__file__))


def data_filepath(filename: str) -> str:
    """Return absolute path to a data file shipped with the project."""
    return os.path.join(BASE_PATH, "data", filename)


def convert_to_local(df: pd.DataFrame) -> pd.DataFrame:
    local_tz = get_localzone()
    assert df.index.tz is not None, "DataFrame index must be timezone aware"
    return df.tz_convert(local_tz)


def ensure_utc(dt: datetime | date) -> datetime:
    """Return a timezone-aware UTC datetime for ``dt``."""
    assert isinstance(dt, (datetime, date)), "dt must be date or datetime"
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    if isinstance(dt, date):
        return datetime.combine(dt, time.min, tzinfo=timezone.utc)
    raise TypeError(f"Unsupported type for ensure_utc: {type(dt)!r}")


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
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
    values: Iterable[Any], *, context: str = "", try_units: tuple[str | None, ...] | None = None
) -> pd.DatetimeIndex:
    """Return a timezone-aware ``DatetimeIndex`` from ``values``.

    Tries multiple ``unit`` formats and raises ``ValueError`` if parsing fails
    for all provided units. ``context`` is included in log messages for easier
    debugging.
    """

    if values is None:
        return pd.DatetimeIndex([], tz=timezone.utc)
    if isinstance(values, str):
        raise ValueError("values must be iterable, not str")
    if isinstance(values, pd.MultiIndex):
        values = values.get_level_values(-1)

    arr = np.asarray(list(values))
    if arr.size == 0 or np.all(pd.isna(arr)):
        _warn_limited(
            "empty-values",
            "[safe_to_datetime]%s: All values missing or empty: %r",
            context,
            list(values)[:5],
        )
        return pd.DatetimeIndex([], tz=timezone.utc)

    if arr.dtype.kind in {"U", "S", "O"}:
        bad = [v for v in arr if isinstance(v, str) and not _DATE_RE.match(v)]
        if bad:
            _warn_limited(
                "invalid-date",
                "[safe_to_datetime]%s: Non-date strings encountered: %r",
                context,
                bad[:5],
            )
            raise ValueError(f"Invalid date strings in {context}: {bad[:5]}")

    ms_first = os.getenv("DEFAULT_MS_EPOCH", "0").lower() in {"1", "true", "yes"}
    if try_units is None:
        try_units = ("ms", "s", None) if ms_first else ("s", "ms", None)

    for unit in try_units:
        try:
            if unit:
                idx = pd.to_datetime(arr, unit=unit, errors="coerce", utc=True)
            else:
                sample = next((s for s in arr if isinstance(s, str)), "")
                fmt = None
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", sample):
                    fmt = "%Y-%m-%d"
                elif re.fullmatch(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", sample):
                    fmt = "%Y-%m-%d %H:%M:%S" if " " in sample else "%Y-%m-%dT%H:%M:%S"
                idx = pd.to_datetime(arr, format=fmt, errors="coerce", utc=True)
        except Exception as e:  # pragma: no cover - unexpected failures
            logger.error(
                "[safe_to_datetime]%s: Failed with unit=%s: %s",
                context,
                unit,
                e,
            )
            continue
        if idx.notna().any():
            if idx.isna().any():
                _warn_limited(
                    "partial-parse",
                    "[safe_to_datetime]%s: Some timestamps failed to parse: %r",
                    context,
                    arr[idx.isna()].tolist(),
                )
            return idx

    logger.error(
        "[safe_to_datetime]%s: All formats failed for values: %r",
        context,
        list(values)[:5],
    )
    raise ValueError(f"Could not parse any timestamps in {context}: {values!r}")

# Generic robust column getter with validation

def get_column(df, options, label, dtype=None, must_be_monotonic=False,
               must_be_non_null=False, must_be_unique=False, must_be_timezone_aware=False):
    for col in options:
        if col in df.columns:
            if dtype is not None:
                if dtype == "datetime64[ns]" and pd.api.types.is_datetime64_any_dtype(df[col]):
                    pass
                elif not pd.api.types.is_dtype_equal(df[col].dtype, dtype):
                    raise TypeError(f"{label}: column '{col}' is not of dtype {dtype}, got {df[col].dtype}")
            if must_be_monotonic and not df[col].is_monotonic_increasing:
                raise ValueError(f"{label}: column '{col}' is not monotonic increasing")
            if must_be_non_null and df[col].isnull().all():
                raise ValueError(f"{label}: column '{col}' is all null")
            if must_be_unique and not df[col].is_unique:
                raise ValueError(f"{label}: column '{col}' is not unique")
            if must_be_timezone_aware and hasattr(df[col], "dt") and df[col].dt.tz is None:
                raise ValueError(f"{label}: column '{col}' is not timezone-aware")
            return col
    raise ValueError(f"No recognized {label} column found in DataFrame: {df.columns.tolist()}")

# OHLCV helpers

def get_open_column(df):
    return get_column(df, ["Open", "open", "o"], "open price", dtype=None)

def get_high_column(df):
    return get_column(df, ["High", "high", "h"], "high price", dtype=None)

def get_low_column(df):
    return get_column(df, ["Low", "low", "l"], "low price", dtype=None)

def get_close_column(df):
    return get_column(df, ["Close", "close", "c", "adj_close", "Adj Close", "adjclose", "adjusted_close"], "close price", dtype=None)

def get_volume_column(df):
    return get_column(df, ["Volume", "volume", "v"], "volume", dtype=None)

# Datetime helper with advanced checks

def _safe_get_column(df, options, label, **kwargs):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    try:
        return get_column(df, options, label, **kwargs)
    except Exception:
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
    return _safe_get_column(df, ["symbol", "ticker", "SYMBOL"], "symbol", dtype="O", must_be_unique=True)

# Return/returns column

def get_return_column(df):
    return _safe_get_column(df, ["Return", "ret", "returns"], "return", dtype=None, must_be_non_null=True)

# Indicator column (pass a list, e.g. ["SMA", "sma", "EMA", ...])

def get_indicator_column(df, possible_names):
    return _safe_get_column(df, possible_names, "indicator")

# Order/trade columns

def get_order_column(df, name):
    return _safe_get_column(df, [name, name.lower(), name.upper()], f"order/{name}", dtype=None, must_be_non_null=True)


def get_ohlcv_columns(df):
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
    except Exception:
        return []
