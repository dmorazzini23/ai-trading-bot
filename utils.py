"""Utility functions for common operations across the bot."""

import datetime as dt
import logging
import os
import socket
import threading
import warnings
import time
from datetime import date, timezone
from typing import Any, Sequence
from enum import Enum
from uuid import UUID
from zoneinfo import ZoneInfo

# AI-AGENT-REF: guard pandas import for test environments
try:
    import pandas as pd
except ImportError:
    # AI-AGENT-REF: pandas not available - create minimal fallbacks for import compatibility
    from datetime import datetime
    class MockDataFrame:
        def __init__(self, *args, **kwargs):
            pass
        def __len__(self):
            return 0
        def empty(self):
            return True
    class MockSeries:
        def __init__(self, *args, **kwargs):
            pass
        def __len__(self):
            return 0
    class MockPandas:
        DataFrame = MockDataFrame
        Series = MockSeries
        Timestamp = datetime  # AI-AGENT-REF: mock Timestamp with datetime
        def read_csv(self, *args, **kwargs):
            return MockDataFrame()
        def concat(self, *args, **kwargs):
            return MockDataFrame()
    pd = MockPandas()
import config
import random

logger = logging.getLogger(__name__)

try:
    import psutil
except ImportError:
    psutil = None
    logger.warning("psutil import failed — memory stats disabled")

try:
    from tzlocal import get_localzone
except ImportError:  # pragma: no cover - optional dependency
    logging.warning("tzlocal not installed; defaulting to UTC")

    def get_localzone() -> ZoneInfo:
        return ZoneInfo("UTC")


# AI-AGENT-REF: throttle noisy logs
_LAST_MARKET_HOURS_LOG = 0.0
_LAST_MARKET_STATE = ""
_LAST_HEALTH_ROW_LOG = 0.0
_LAST_HEALTH_ROWS_COUNT = -1
_LAST_HEALTH_STATUS: bool | None = None

# AI-AGENT-REF: throttle HEALTH_ROWS logs

warnings.filterwarnings("ignore", category=FutureWarning)


class PhaseLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that injects bot_phase context."""

    def process(self, msg, kwargs):
        extra = kwargs.setdefault("extra", {})
        extra.setdefault("bot_phase", self.extra.get("bot_phase", "GENERAL"))
        extra.setdefault("timestamp", dt.datetime.now(timezone.utc))
        return msg, kwargs


def get_phase_logger(name: str, phase: str) -> logging.Logger:
    """Return logger with ``bot_phase`` context."""
    base = logging.getLogger(name)
    return PhaseLoggerAdapter(base, {"bot_phase": phase})


def log_cpu_usage(lg: logging.Logger, note: str | None = None) -> None:
    """Log current process CPU usage if :mod:`psutil` is available."""
    if psutil is None:
        return
    pct = psutil.cpu_percent(interval=None)
    suffix = f"_{note}" if note else ""
    lg.debug("CPU_USAGE%s: %.2f%%", suffix, pct)


MIN_HEALTH_ROWS = int(os.getenv("MIN_HEALTH_ROWS", "30"))
MIN_HEALTH_ROWS_D = int(os.getenv("MIN_HEALTH_ROWS_DAILY", "5"))
HEALTH_MIN_ROWS = int(os.getenv("HEALTH_MIN_ROWS", "100"))
HEALTH_THROTTLE = 10
_last_health_log = 0.0


def log_warning(
    msg: str, *, exc: Exception | None = None, extra: dict | None = None
) -> None:
    """Standardized warning logger used across the project."""
    if extra is None:
        extra = {}
    if exc is not None:
        if msg == "HEALTH_STALE_DATA":
            logger.debug("%s: %s", msg, exc, extra=extra, exc_info=True)
        else:
            logger.warning("%s: %s", msg, exc, extra=extra, exc_info=True)
    else:
        if msg == "HEALTH_STALE_DATA":
            logger.debug(msg, extra=extra)
        else:
            logger.warning(msg, extra=extra)


# Cache of last logged stale timestamp per symbol
_STALE_CACHE: dict[str, tuple[pd.Timestamp, float]] = {}
# AI-AGENT-REF: Add thread-safe locking for cache operations
_STALE_CACHE_LOCK = threading.Lock()


def should_log_stale(symbol: str, last_ts: pd.Timestamp, *, ttl: int = 300) -> bool:
    """Check if stale data warning should be logged for this symbol."""
    import time
    current_time = time.time()

    # AI-AGENT-REF: Add thread-safe locking for cache operations
    with _STALE_CACHE_LOCK:
        if symbol in _STALE_CACHE:
            cached_ts, cached_time = _STALE_CACHE[symbol]
            if cached_ts == last_ts and (current_time - cached_time) < ttl:
                return False

        _STALE_CACHE[symbol] = (last_ts, current_time)
        return True

def backoff_delay(attempt: int, base: float = 1.0, cap: float = 30.0, jitter: float = 0.1) -> float:
    """Return exponential backoff delay with jitter."""
    exp = base * (2 ** max(0, attempt - 1))
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
        if isinstance(v, (dt.datetime, date)):
            val = v.isoformat()
        elif isinstance(v, Enum):
            val = v.value
        elif isinstance(v, UUID):
            val = str(v)
        elif isinstance(v, (int, float, bool)) or v is None:
            val = v
        else:
            val = str(v)
        parts.append(f"{k}={val}")
    return ", ".join(parts)


MARKET_OPEN_TIME = dt.time(9, 30)
MARKET_CLOSE_TIME = dt.time(16, 0)
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
    """Return last closing price or ``0.0`` if unavailable."""
    if df is None or df.empty:
        return 0.0
    if "close" not in df.columns:
        return 0.0
    last_valid_close = df["close"].dropna()
    if not last_valid_close.empty:
        price = last_valid_close.iloc[-1]
    else:
        logger.critical("All NaNs in close column for get_latest_close")
        price = 0.0
    if pd.isna(price) or price <= 0:
        return 0.0
    return float(price)


def get_current_price(symbol: str) -> float:
    """Return latest quote price with fallbacks."""
    price = 0.0
    try:
        from alpaca_api import alpaca_get

        data = alpaca_get(f"/v2/stocks/{symbol}/quotes/latest")
        price = float(data.get("ap", 0) or 0)
    except Exception as exc:  # pragma: no cover - network/API errors
        logger.warning("get_current_price primary fetch failed for %s: %s", symbol, exc)

    if price <= 0:
        logger.warning(
            "get_current_price invalid price %.2f for %s; falling back to last close",
            price,
            symbol,
        )
        try:
            from data_fetcher import get_daily_df

            end = dt.date.today()
            start = end - dt.timedelta(days=5)
            df = get_daily_df(symbol, start, end)
            price = get_latest_close(df) if df is not None else 0.0
        except Exception as exc:  # pragma: no cover - fallback errors
            logger.warning("get_current_price fallback failed for %s: %s", symbol, exc)

    if price <= 0:
        logger.warning("get_current_price ultimate fallback using 0.01 for %s", symbol)
        price = 0.01
    return price


def _log_market_hours(message: str) -> None:
    """Emit market hours message only on state change or hourly."""
    global _LAST_MARKET_HOURS_LOG, _LAST_MARKET_STATE
    now = time.time()
    state = "OPEN" if "OPEN" in message else "CLOSED"
    if state != _LAST_MARKET_STATE or now - _LAST_MARKET_HOURS_LOG >= 3600:
        if config.VERBOSE_LOGGING:
            logger.info(message)
        else:
            logger.debug(message)
        _LAST_MARKET_STATE = state
        _LAST_MARKET_HOURS_LOG = now


def log_health_row_check(rows: int, passed: bool) -> None:
    """Log HEALTH_ROWS status changes or once every 10 seconds."""
    global _LAST_HEALTH_ROW_LOG, _LAST_HEALTH_ROWS_COUNT, _LAST_HEALTH_STATUS
    now = time.monotonic()
    if (
        not passed
        or rows != _LAST_HEALTH_ROWS_COUNT
        or passed != _LAST_HEALTH_STATUS
        or now - _LAST_HEALTH_ROW_LOG >= 10
    ):
        level = logger.info if config.VERBOSE_LOGGING or not passed else logger.debug
        status = "PASSED" if passed else "FAILED"
        level("HEALTH_ROWS_%s: received %d rows", status, rows)
        _LAST_HEALTH_ROW_LOG = now
        _LAST_HEALTH_ROWS_COUNT = rows
        _LAST_HEALTH_STATUS = passed


def health_rows_passed(rows):
    """Log HEALTH_ROWS_PASSED with throttling."""
    global _last_health_log
    now = time.monotonic()
    if _last_health_log == 0.0 or now - _last_health_log >= HEALTH_THROTTLE:
        count = len(rows) if not isinstance(rows, (int, float)) else rows
        logger.debug("HEALTH_ROWS_PASSED: received %d rows", count)
        _last_health_log = now or 1e-9
    else:
        logger.debug("HEALTH_ROWS_THROTTLED")
    return rows


def is_market_open(now: dt.datetime | None = None) -> bool:
    """Return True if current time is within NYSE trading hours."""
    if os.getenv("FORCE_MARKET_OPEN", "false").lower() == "true":
        logger.info("FORCE_MARKET_OPEN is enabled; overriding market hours checks.")
        return True
    try:
        import pandas_market_calendars as mcal

        check_time = (now or dt.datetime.now(timezone.utc)).astimezone(EASTERN_TZ)
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
            _log_market_hours("Detected Market Hours today: CLOSED")
            return False  # holiday or weekend
        market_open = sched.iloc[0]["market_open"].tz_convert(EASTERN_TZ).time()
        market_close = sched.iloc[0]["market_close"].tz_convert(EASTERN_TZ).time()
        if check_time.month == 7 and check_time.day == 3:
            july4 = date(check_time.year, 7, 4)
            if july4.weekday() >= 5:
                market_close = time(13, 0)
            else:
                market_close = MARKET_CLOSE_TIME
        _log_market_hours(
            "Detected Market Hours today: OPEN from %s to %s"
            % (market_open.strftime("%H:%M"), market_close.strftime("%H:%M"))
        )
        current = check_time.time()
        return market_open <= current <= market_close
    except Exception as exc:
        logger.debug("market calendar unavailable: %s", exc)
        # Fallback to simple weekday/time check when calendar unavailable
        now_et = (now or dt.datetime.now(timezone.utc)).astimezone(EASTERN_TZ)
        if now_et.weekday() >= 5:
            _log_market_hours("Detected Market Hours today: CLOSED")
            return False
        current = now_et.time()
        _log_market_hours(
            "Detected Market Hours today: OPEN from %s to %s"
            % (
                MARKET_OPEN_TIME.strftime("%H:%M"),
                MARKET_CLOSE_TIME.strftime("%H:%M"),
            )
        )
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
        return dt.datetime.combine(value, dt.time.min, tzinfo=timezone.utc)
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


def _pid_from_inode(inode: str) -> int | None:
    """Return PID for a socket inode on Linux."""
    for pid in filter(str.isdigit, os.listdir("/proc")):
        fd_dir = f"/proc/{pid}/fd"
        if not os.path.isdir(fd_dir):
            continue
        for fd in os.listdir(fd_dir):
            try:
                if os.readlink(os.path.join(fd_dir, fd)) == f"socket:[{inode}]":
                    return int(pid)
            except OSError:
                continue
    return None


def get_pid_on_port(port: int) -> int | None:
    """Best-effort detection of PID bound to ``port``."""
    try:
        with open("/proc/net/tcp") as f:
            next(f)
            for line in f:
                parts = line.split()
                local = parts[1]
                inode = parts[9]
                if int(local.split(":")[1], 16) == port:
                    return _pid_from_inode(inode)
    except Exception as e:
        logging.getLogger(__name__).error("get_pid_on_port failed", exc_info=e)
        return None
    return None


def get_rolling_atr(symbol: str, window: int = 14) -> float:
    """Return normalized ATR over ``window`` days."""
    from bot_engine import fetch_minute_df_safe  # lazy import to avoid cycles

    df = fetch_minute_df_safe(symbol)
    if df is None or df.empty:
        return 0.0
    high = df["high"].rolling(window).max()
    low = df["low"].rolling(window).min()
    close = df["close"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
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
    from bot_engine import fetch_minute_df_safe

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
    from bot_engine import fetch_minute_df_safe

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
        from ml_model import load_model
    except Exception as e:  # pragma: no cover - optional dependency
        logger.error("load_model failed", exc_info=e)
        return 0.5

    model_path = config.MODEL_PATH
    model = load_model(model_path)
    if model is None:
        return 0.5
    feats = pd.DataFrame({"price": [0.0]})
    try:
        conf = float(model.predict_proba(feats)[0][1])
    except Exception as e:
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

    # Prefer a fast path when the format matches exactly.  If parsing fails
    # (for example when encountering numeric placeholders like "0") fall back to
    # a more permissive conversion.  This two-stage approach avoids spurious
    # warnings while still handling heterogeneous input gracefully.
    try:
        return pd.to_datetime(arr, format=format, utc=utc)
    except (TypeError, ValueError) as exc:
        # Log once per unique context key to avoid log spam; include context if set
        ctx = f" ({context})" if context else ""
        logger.warning(
            "safe_to_datetime coercing invalid values%s – %s", ctx, exc
        )
        try:
            # Let pandas infer the format and coerce invalid entries to NaT
            return pd.to_datetime(arr, errors="coerce", utc=utc)
        except Exception as exc2:
            logger.error("safe_to_datetime failed%s: %s", ctx, exc2)
            # Fall back to an array of NaT with proper timezone
            length = len(arr) if hasattr(arr, "__len__") else 1
            return pd.DatetimeIndex([pd.NaT] * length, tz="UTC")


def health_check(df: pd.DataFrame | None, resolution: str) -> bool:
    """Validate that ``df`` has enough rows for reliable analysis."""
    min_rows = int(os.getenv("HEALTH_MIN_ROWS", 100))

    if df is None:
        logger.critical("HEALTH_FAILURE: DataFrame is None.")
        return False

    rows = len(df)
    if rows < min_rows:
        logger.warning(
            "HEALTH_INSUFFICIENT_ROWS: only %d rows (min expected %d)",
            rows,
            min_rows,
        )
        logger.debug("Shape: %s", df.shape)
        logger.debug("Columns: %s", df.columns.tolist())
        logger.debug("Preview:\n%s", df.head(3))
        if rows == 0:
            logger.critical("HEALTH_FAILURE: empty dataset loaded")
        log_health_row_check(rows, False)
        return False

    log_health_row_check(rows, True)
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
    return _safe_get_column(df, ["Open", "open", "o"], "open price", dtype=None)


def get_high_column(df):
    return _safe_get_column(df, ["High", "high", "h"], "high price", dtype=None)


def get_low_column(df):
    return _safe_get_column(df, ["Low", "low", "l"], "low price", dtype=None)


def get_close_column(df):
    return _safe_get_column(
        df,
        ["Close", "close", "c", "adj_close", "Adj Close", "adjclose", "adjusted_close"],
        "close price",
        dtype=None,
    )


def get_volume_column(df):
    return _safe_get_column(df, ["Volume", "volume", "v"], "volume", dtype=None)


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
    cols = [
        get_open_column(df),
        get_high_column(df),
        get_low_column(df),
        get_close_column(df),
        get_volume_column(df),
    ]
    if any(c is None for c in cols):
        return []
    return cols


REQUIRED_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


def validate_ohlcv(df: pd.DataFrame) -> bool:
    """Return True if ``df`` contains the required OHLCV columns."""

    if not isinstance(df, pd.DataFrame) or df.empty:
        logger.error("validate_ohlcv received invalid DataFrame")
        return False
    missing = [c for c in REQUIRED_OHLCV_COLS if c not in df.columns]
    if missing:
        logger.error("Missing OHLCV columns: %s", missing)
        return False
    return True


from typing import Dict, List

try:
    from alpaca_trade_api.rest import REST
except Exception as e:  # pragma: no cover - optional dependency
    logging.getLogger(__name__).error("alpaca_trade_api import failed", exc_info=e)
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
            logging.warning(f"Health check skipped for {sym}: insufficient data")
    return symbol_health
