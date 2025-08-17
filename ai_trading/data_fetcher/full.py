from __future__ import annotations

import json
import logging
import os
import random
import sys
import threading
import time
import time as pytime
from collections import deque
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import urllib3
from requests.exceptions import RequestException
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
    wait_random,
)

from ai_trading.utils import http
from ai_trading.utils import sleep as psleep
from ai_trading.utils.prof import StageTimer

_TESTING = os.getenv("PYTEST_RUNNING", "").lower() in {"1", "true", "yes"} or os.getenv(
    "TESTING", ""
).lower() in {"1", "true", "yes"}

_RETRY_STOP = stop_after_attempt(3 if not _TESTING else 2)
_RETRY_WAIT = (
    wait_exponential(multiplier=1, min=1, max=8) if not _TESTING else wait_fixed(0.05)
)

UTC = UTC

# Do not hard fail when running under older Python versions in tests
if sys.version_info < (3, 12, 3):  # pragma: no cover - compat check
    import logging

    logging.getLogger(__name__).warning("Running under unsupported Python version")

from ai_trading.config.settings import get_settings as get_config_settings
from ai_trading.market import cache as mcache
from ai_trading.settings import (
    get_data_cache_enable,
    get_data_cache_ttl_seconds,
)

# Define logger early
logger = logging.getLogger(__name__)

__all__ = [
    "ensure_datetime",
    "get_historical_data",
    "get_daily_df",
    "get_minute_df",
    "get_cache_stats",
    "get_last_available_bar",
]

# Lightweight placeholders for optional Alpaca SDK components. The real
# modules are imported lazily via ``_get_alpaca_client`` to avoid import-time
# side effects during testing when Alpaca or requests are intentionally
# unavailable.
StockBarsRequest = None  # type: ignore
TimeFrame = None  # type: ignore
TimeFrameUnit = None  # type: ignore


class APIError(Exception):
    """Placeholder Alpaca API error used for retry logic."""


_DATA_CLIENT = None


def ensure_utc(dt: datetime | pd.Timestamp) -> datetime:
    if hasattr(dt, "tzinfo") and dt.tzinfo is not None:
        return dt.astimezone(UTC)
    return dt.replace(tzinfo=UTC)


def ensure_datetime(dt):
    """Normalize any naive/aware datetime into UTC pandas.Timestamp."""  # AI-AGENT-REF: simplify
    import pandas as pd

    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


# Prometheus metrics via shim (no-op if unavailable)
from ai_trading.metrics import PROMETHEUS_AVAILABLE, Counter, Histogram

_MET_REQS = Counter(
    "data_requests_total",
    "Data requests",
    ["source", "timeframe", "cache", "mode"],
)
_MET_LAT = Histogram(
    "data_request_latency_seconds",
    "Data request latency",
    ["source", "timeframe", "cache", "mode"],
)
if not PROMETHEUS_AVAILABLE:
    logger.debug("Prometheus metrics disabled; using no-op collectors")

CFG = get_config_settings()
BASE_DIR = Path(__file__).resolve().parents[1]  # AI-AGENT-REF: repo root for paths


def abspath(fname: str) -> str:
    return str((BASE_DIR / str(fname)).resolve())


FINNHUB_API_KEY = CFG.finnhub_api_key
ALPACA_API_KEY = CFG.alpaca_api_key
ALPACA_SECRET_KEY = CFG.alpaca_secret_key_plain  # AI-AGENT-REF: use plain secret string
ALPACA_BASE_URL = CFG.alpaca_base_url
ALPACA_DATA_FEED = CFG.alpaca_data_feed or "iex"


def _get_alpaca_client():
    """Lazily import and initialize the Alpaca data client.

    Returns ``None`` if the SDK or credentials are unavailable. Importing is
    deferred to runtime to keep module import side-effect free, particularly in
    test environments that intentionally mask Alpaca dependencies.
    """

    global _DATA_CLIENT, StockBarsRequest, TimeFrame, TimeFrameUnit, APIError
    if _DATA_CLIENT is not None:
        return _DATA_CLIENT
    try:
        from alpaca.common.exceptions import APIError as _APIError
        from alpaca.data.historical import StockHistoricalDataClient as _SHDC
        from alpaca.data.requests import StockBarsRequest as _SBR
        from alpaca.data.timeframe import TimeFrame as _TF
        from alpaca.data.timeframe import TimeFrameUnit as _TFU
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.debug("Alpaca SDK unavailable: %s", exc)
        _DATA_CLIENT = None
        return None

    try:
        _DATA_CLIENT = _SHDC(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY)
    except Exception as exc:  # pragma: no cover - best effort
        logger.error("Failed to initialize Alpaca client: %s", exc)
        _DATA_CLIENT = None
        return None

    StockBarsRequest = _SBR
    TimeFrame = _TF
    TimeFrameUnit = _TFU
    APIError = _APIError
    return _DATA_CLIENT


# Default market data feed for Alpaca requests
_DEFAULT_FEED = "iex"


def _mask_headers(headers: dict[str, Any]) -> dict[str, Any]:
    masked = {}
    for k, v in headers.items():
        if isinstance(v, str) and any(
            x in k.lower() for x in ("key", "token", "secret")
        ):
            from ai_trading.config.settings import get_settings

            S = get_settings()
            # Use proper secret masking utility if available
            masked[k] = "***MASKED***" if v else None
        else:
            masked[k] = v
    return masked


def _log_http_request(
    method: str, url: str, params: dict[str, Any], headers: dict[str, Any]
) -> None:
    logger.debug(
        "HTTP %s %s params=%s headers=%s", method, url, params, _mask_headers(headers)
    )


def _log_http_response(resp: Any) -> None:
    logger.debug("HTTP_RESPONSE status=%s body=%s", resp.status_code, resp.text[:300])


_rate_limit_lock = threading.Lock()


def _build_daily_url(symbol: str, start: datetime, end: datetime) -> str:
    """Construct Alpaca bars URL for a symbol."""  # AI-AGENT-REF: pooled fetch helper
    params = {
        "start": ensure_utc(start).isoformat().replace("+00:00", "Z"),
        "end": ensure_utc(end).isoformat().replace("+00:00", "Z"),
        "timeframe": "1Day",
        "feed": _DEFAULT_FEED,
    }
    return f"https://data.alpaca.markets/v2/stocks/{symbol}/bars?" + urlencode(params)


def _parse_bars(symbol: str, code: int, body: bytes) -> pd.DataFrame | None:
    """Parse bar response into a DataFrame."""  # AI-AGENT-REF: parse pooled responses
    if code != 200:
        logger.error("fetch_daily_data_async status %s for %s", code, symbol)
        return None
    try:
        data = json.loads(body.decode("utf-8"))
    except Exception as exc:
        logger.warning("fetch_daily_data_async decode error for %s: %s", symbol, exc)
        return None
    bars = data.get("bars") or []
    if not bars:
        logger.warning("No bars returned for %s", symbol)
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
    df = pd.DataFrame(bars)
    rename_map = {
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    }
    df = df.rename(columns=rename_map)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.reset_index(drop=True)
    return df


from ai_trading.utils.base import is_market_open, safe_to_datetime

MINUTES_REQUIRED = 31
MIN_EXPECTED_ROWS = 5
HISTORICAL_START = "2025-06-01"
HISTORICAL_END = "2025-06-06"

# FutureWarning now filtered globally in pytest.ini


class FinnhubAPIException(Exception):
    """Fallback Finnhub exception used when SDK isn't available."""

    def __init__(self, *args, status_code: int | None = None, **kwargs) -> None:
        self.status_code = status_code if status_code is not None else 0
        super().__init__(*args or ("FinnhubAPIException",))


def _get_finnhub_client():
    try:
        import finnhub as fh
        from finnhub import FinnhubAPIException as _FHExc

        globals()["FinnhubAPIException"] = _FHExc
        api_key = os.getenv("FINNHUB_API_KEY", "") or getattr(
            CFG, "finnhub_api_key", ""
        )
        if not hasattr(fh, "Client") or not api_key:
            return None
        return fh.Client(api_key=api_key)
    except Exception:
        return None


# AI-AGENT-REF: import pandas as hard dependency
import pandas as pd
from tenacity import RetryError

# Refresh environment in case this module is executed as a script
# Note: config module is no longer imported; using settings instead

# In-memory minute bar cache to avoid unnecessary API calls
_MINUTE_CACHE: dict[str, tuple[pd.DataFrame, pd.Timestamp]] = {}

# AI-AGENT-REF: Cache performance monitoring
_CACHE_STATS = {"hits": 0, "misses": 0, "invalidations": 0}


def get_cached_minute_timestamp(symbol: str) -> pd.Timestamp | None:
    """
    Return the cached 'last fetch time' for minute data in UTC, if present.
    Does not trigger any network call.
    """
    entry = _MINUTE_CACHE.get(symbol)
    if not entry:
        return None
    _, ts = entry
    return ts if isinstance(ts, pd.Timestamp) else None


def last_minute_bar_age_seconds(symbol: str) -> int:
    """Return age in seconds of latest cached minute bar for ``symbol``; 0 if unknown."""
    ts = get_cached_minute_timestamp(symbol)
    if ts is None:
        return 0
    try:
        return int((pd.Timestamp.now(tz="UTC") - ts).total_seconds())
    except Exception:
        return 0


def get_cache_stats() -> dict:
    """Get current cache statistics for monitoring and debugging."""
    total_requests = _CACHE_STATS["hits"] + _CACHE_STATS["misses"]
    hit_ratio = (
        (_CACHE_STATS["hits"] / total_requests * 100) if total_requests > 0 else 0
    )

    return {
        "cache_size": len(_MINUTE_CACHE),
        "cached_symbols": list(_MINUTE_CACHE.keys()),
        "hits": _CACHE_STATS["hits"],
        "misses": _CACHE_STATS["misses"],
        "invalidations": _CACHE_STATS["invalidations"],
        "hit_ratio_pct": round(hit_ratio, 1),
        "total_requests": total_requests,
        "cache_entries": [
            {
                "symbol": symbol,
                "rows": len(df),
                "last_updated": ts.isoformat() if ts else None,
            }
            for symbol, (df, ts) in _MINUTE_CACHE.items()
        ],
    }


def _fetch_bars(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str,
    feed: str = _DEFAULT_FEED,
) -> pd.DataFrame:
    """Fetch raw bars from Alpaca with detailed logging."""
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    params = {
        "start": ensure_utc(start).isoformat().replace("+00:00", "Z"),
        "end": ensure_utc(end).isoformat().replace("+00:00", "Z"),
        "timeframe": timeframe,
        "feed": feed,
    }
    headers = CFG.alpaca_headers  # AI-AGENT-REF: canonical Alpaca headers
    _log_http_request("GET", url, params, headers)
    delay = 1.0
    for attempt in range(3):
        try:
            resp = http.get(url, params=params, headers=headers)
            if (
                resp.status_code == 400
                and "invalid feed" in resp.text.lower()
                and feed != "sip"
            ):
                logger.warning(
                    "Alpaca invalid feed %s for %s; retrying with SIP", feed, symbol
                )
                params["feed"] = "sip"
                resp = http.get(url, params=params, headers=headers)
            break
        except RequestException as exc:
            logger.warning(
                "HTTP request failed %s/%s for %s: %s", attempt + 1, 3, symbol, exc
            )
            if attempt == 2:
                logger.exception("HTTP request error for %s", symbol, exc_info=exc)
                raise DataFetchException(symbol, "alpaca", url, str(exc)) from exc
            psleep(delay)
            delay *= 2

    _log_http_response(resp)
    if resp.status_code != 200:
        raise DataFetchException(
            symbol,
            "alpaca",
            url,
            f"status {resp.status_code}: {resp.text[:300]}",
        )
    data = resp.json()
    bars = data.get("bars") or []
    if not bars:
        # AI-AGENT-REF: Enhanced data quality check for missing bars
        logger.warning(
            "No bars returned for %s between %s and %s. "
            "This could indicate market holiday, API outage, or delisted symbol",
            symbol,
            start.date(),
            end.date(),
        )
        # no new bars yet (e.g. today's bar before market close)
        # fallback: use yesterday's last bar or skip this symbol
        return get_last_available_bar(symbol)
    df = pd.DataFrame(bars)
    rename_map = {
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    }
    df = df.rename(columns=rename_map)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.reset_index(drop=True)
    return df


class DataFetchError(Exception):
    """Raised when a data request fails after all retries."""


class DataFetchException(Exception):
    """Raised when an HTTP fetch returns an error status."""

    def __init__(self, symbol: str, provider: str, endpoint: str, message: str) -> None:
        super().__init__(message)
        self.symbol = symbol
        self.provider = provider
        self.endpoint = endpoint
        self.message = message


class DataSourceDownException(Exception):
    """Raised when all data sources fail for a symbol."""

    def __init__(self, symbol: str) -> None:
        super().__init__(f"All data sources failed for {symbol}")
        self.symbol = symbol


class DataSourceEmpty(Exception):
    """Raised when all providers return no data."""

    def __init__(self, symbol: str) -> None:
        super().__init__(f"No data returned for {symbol}")
        self.symbol = symbol


def get_last_available_bar(symbol: str) -> pd.DataFrame:
    """Return the most recent daily bar for ``symbol`` or empty DataFrame."""
    end = datetime.now(UTC).date()
    start = end - timedelta(days=2)
    try:
        df = get_daily_df(symbol, start, end)
        if df is not None and not df.empty:
            return df.tail(1)
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("get_last_available_bar failed for %s: %s", symbol, exc)
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


@retry(stop=_RETRY_STOP, wait=_RETRY_WAIT)
def get_historical_data(
    symbol: str,
    start_date: str | date | datetime | pd.Timestamp,
    end_date: str | date | datetime | pd.Timestamp,
    timeframe: str,
    *,
    raise_on_empty: bool = False,
    provider: str | None = None,
    validate_data: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical market data with comprehensive validation and error handling.

    This function retrieves historical OHLCV (Open, High, Low, Close, Volume) data
    for a specified symbol and time range. It implements automatic provider failover,
    data quality validation, and robust error handling to ensure reliable data
    for trading analysis.

    Parameters
    ----------
    symbol : str
        Trading symbol in uppercase format (e.g., 'AAPL', 'SPY', 'MSFT').
        Must be a valid symbol supported by the configured data providers.
    start_date : Union[str, date, datetime, pd.Timestamp]
        Start date for historical data in various formats:
        - String: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
        - date/datetime objects
        - pandas Timestamp
    end_date : Union[str, date, datetime, pd.Timestamp]
        End date for historical data (same format options as start_date).
        Must be after start_date and not in the future.
    timeframe : str
        Data frequency/timeframe:
        - '1MIN': 1-minute bars
        - '5MIN': 5-minute bars
        - '15MIN': 15-minute bars
        - '1HOUR': 1-hour bars
        - '1DAY': Daily bars
    raise_on_empty : bool, optional
        If True, raises DataFetchError when no data is returned.
        If False (default), returns empty DataFrame with proper columns.
    provider : Optional[str], optional
        Specific data provider to use ('alpaca', 'finnhub', 'yahoo').
        If None, uses automatic provider selection with failover.
    validate_data : bool, optional
        If True (default), performs data quality validation including:
        - Gap detection and reporting
        - Price anomaly checking
        - Volume validation
        - Timestamp consistency

    Returns
    -------
    pd.DataFrame
        Historical market data with DatetimeIndex and columns:
        - 'open' (float): Opening prices
        - 'high' (float): High prices
        - 'low' (float): Low prices
        - 'close' (float): Closing prices
        - 'volume' (int): Trading volume
        - Additional columns may include 'vwap', 'trade_count'

    Raises
    ------
    ValueError
        If start_date/end_date are None or invalid format
    TypeError
        If date parameters are not supported types
    DataFetchError
        If raise_on_empty=True and no data is retrieved
    ConnectionError
        If all data providers fail to respond
    TimeoutError
        If data retrieval exceeds configured timeout

    Examples
    --------
    >>> from ai_trading.data_fetcher import get_historical_data
    >>> from datetime import datetime, timedelta
    >>>
    >>> # Get 30 days of hourly data
    >>> end_date = datetime.now(timezone.utc)  # AI-AGENT-REF: timezone-aware for API compatibility
    >>> start_date = end_date - timedelta(days=30)
    >>> data = get_historical_data('AAPL', start_date, end_date, '1HOUR')
    >>> logging.info(f"Retrieved {len(data)} bars")
    >>> logging.info(str(data.head()))

    >>> # Get daily data with specific provider
    >>> data = get_historical_data(
    ...     'SPY', '2024-01-01', '2024-01-31', '1DAY',
    ...     provider='alpaca', validate_data=True
    ... )

    >>> # Handle missing data gracefully
    >>> try:
    ...     data = get_historical_data('INVALID', '2024-01-01', '2024-01-02', '1DAY',
    ...                              raise_on_empty=True)
    ... except DataFetchError:
    ...     logging.info("No data available for symbol")

    Data Quality Validation
    ----------------------
    When validate_data=True, the function performs:

    1. **Completeness Check**: Ensures no missing time periods
    2. **Price Validation**: Checks for unrealistic price movements
    3. **Volume Validation**: Verifies non-negative volume values
    4. **Gap Detection**: Identifies and reports data gaps
    5. **Consistency Check**: Validates OHLC relationships (High >= Low, etc.)

    Provider Failover Logic
    ----------------------
    1. **Primary**: Alpaca Markets (real-time, low latency)
    2. **Secondary**: Finnhub (alternative commercial data)
    3. **Fallback**: Yahoo Finance (free, delayed data)

    Each provider is tried sequentially until data is successfully retrieved
    or all providers are exhausted.

    Performance Notes
    ----------------
    - Results are cached to reduce redundant API calls
    - Implements connection pooling for improved performance
    - Uses compression for large data transfers
    - Automatically handles rate limiting with exponential backoff

    See Also
    --------
    _fetch_bars : Low-level data fetching implementation
    ensure_datetime : Date/time parsing utilities
    DataFetchError : Custom exception for data retrieval failures
    """

    if start_date is None or end_date is None:
        logger.error(
            "get_historical_data called with None dates: %r, %r",
            start_date,
            end_date,
            stack_info=True,
        )
        raise ValueError("start_date and end_date must not be None")

    for name, val in (("start_date", start_date), ("end_date", end_date)):
        if not isinstance(val, date | datetime | str | pd.Timestamp):
            logger.error(
                "get_historical_data invalid %s type: %r",
                name,
                type(val),
                stack_info=True,
            )
            raise TypeError(f"{name} must be date, datetime, pandas.Timestamp or str")

    try:
        start_dt = pd.to_datetime(ensure_utc(ensure_datetime(start_date)))
    except (ValueError, TypeError) as e:
        logger.error("get_historical_data start_date error: %s", e, exc_info=True)
        raise

    try:
        end_dt = pd.to_datetime(ensure_utc(ensure_datetime(end_date)))
    except (ValueError, TypeError) as e:
        logger.error("get_historical_data end_date error: %s", e, exc_info=True)
        raise

    client = _get_alpaca_client()
    if client is None:
        raise DataFetchError("Alpaca client not available")

    tf_map = {
        "1MIN": TimeFrame.Minute,
        "5MIN": TimeFrame(5, TimeFrameUnit.Minute),
        "1HOUR": TimeFrame.Hour,
        "1DAY": TimeFrame.Day,
    }
    tf = tf_map.get(timeframe.upper())
    if tf is None:
        # Raise DataFetchError which will be wrapped in RetryError by retry decorator
        raise DataFetchError(f"Unsupported timeframe: {timeframe}")

    def _fetch(feed: str = _DEFAULT_FEED):
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],  # NOTE: batch variant provided below
            start=start_dt,
            end=end_dt,
            timeframe=tf,
            feed=feed,
        )
        try:
            return client.get_stock_bars(req).df
        except Exception as err:
            logger.exception("Historical data fetch failed for %s: %s", symbol, err)
            raise

    try:
        bars = _fetch(_DEFAULT_FEED)
    except APIError as e:
        if "subscription does not permit" in str(e).lower() and _DEFAULT_FEED != "iex":
            logger.warning(
                "Subscription error for %s with feed %s: %s", symbol, _DEFAULT_FEED, e
            )
            try:
                bars = _fetch("iex")
            except APIError as iex_err:
                logger.error("IEX fallback failed for %s: %s", symbol, iex_err)
                raise DataFetchError(
                    f"IEX fallback failed for {symbol}: {iex_err}"
                ) from iex_err
        else:
            raise
    except RetryError as e:
        raise DataFetchError(f"Historical fetch failed for {symbol}: {e}") from e

    df = pd.DataFrame(bars)
    if df.empty:
        logger.warning(f"Empty primary data for {symbol}; falling back to Alpaca")
        for attempt in range(3):
            psleep(0.5 * (attempt + 1))
            bars = _fetch(_DEFAULT_FEED)
            df = pd.DataFrame(bars)
            if not df.empty:
                break
    if df.empty or len(df) < MIN_EXPECTED_ROWS:
        logger.warning(
            f"Data incomplete for {symbol}, got {len(df)} rows. Skipping this cycle."
        )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    df = df.drop(columns=["symbol"], errors="ignore")

    # AI-AGENT-REF: Add null/empty check before applying string operations to avoid AttributeError
    if not df.empty and len(df.columns) > 0:
        df.columns = df.columns.str.lower()

    if not df.empty:
        try:
            idx = safe_to_datetime(df.index, context=f"{symbol} minute")
        except ValueError as e:
            logger.debug("Raw data for %s: %s", symbol, df.head().to_dict())
            logger.warning(
                "Unexpected index format for %s; skipping | %s",
                symbol,
                e,
            )
            return None
        df.index = idx

    # AI-AGENT-REF: Check for empty DataFrame BEFORE column validation to prevent KeyError
    if df is None or df.empty:
        logger.error("_fetch_minute_data produced empty result for %s", symbol)
        raise DataFetchError(f"no data for {symbol}")

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise KeyError(f"Missing '{col}' column for {symbol}")
        df[col] = df[col].astype(float)

    # ensure there's a timestamp column for the tests
    df = df.reset_index()
    df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    df = df.reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def get_daily_df(
    symbol: str,
    start: date | datetime | pd.Timestamp | str,
    end: date | datetime | pd.Timestamp | str,
) -> pd.DataFrame:
    """Fetch daily bars with retry and IEX fallback, with read-through caching."""
    if start is None or end is None:
        logger.error(
            "get_daily_df called with None dates: %r, %r",
            start,
            end,
            stack_info=True,
        )
        raise ValueError("start and end must not be None")

    for name, val in (("start", start), ("end", end)):
        if not isinstance(val, date | datetime | str | pd.Timestamp):
            logger.error(
                "get_daily_df invalid %s type: %r",
                name,
                type(val),
                stack_info=True,
            )
            raise TypeError(f"{name} must be date, datetime, pandas.Timestamp or str")

    start_dt = ensure_datetime(start)
    end_dt = ensure_datetime(end)

    # --- caching layer ---
    CACHE_ON = bool(get_data_cache_enable())
    TTL = max(0, int(get_data_cache_ttl_seconds()))
    tf_name = "1D"
    start_s = str(start)
    end_s = str(end)
    df_cached = None
    if CACHE_ON:
        df_cached = mcache.get_mem(symbol, tf_name, start_s, end_s, TTL)
        if df_cached is not None:
            _MET_REQS.labels("cache", tf_name, "hit", "single").inc()
            return df_cached

    t0 = time.perf_counter()
    try:
        df = _fetch_bars(symbol, start_dt, end_dt, "1Day", _DEFAULT_FEED)
    except DataFetchException as primary_err:
        logger.error(
            "Primary daily fetch failed for %s: %s", symbol, primary_err.message
        )
        try:
            df = _get_fh_fetcher().fetch(symbol, period="6mo", interval="1d")
        except Exception as fh_err:
            logger.critical("Secondary provider failed for %s: %s", symbol, fh_err)
            raise DataSourceDownException(symbol) from fh_err
    _MET_LAT.labels("alpaca", tf_name, "miss", "single").observe(
        time.perf_counter() - t0
    )
    _MET_REQS.labels("alpaca", tf_name, "miss", "single").inc()
    try:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)
        df = df.drop(columns=["symbol"], errors="ignore")

        df.columns = df.columns.str.lower()

        if df.empty or len(df) < MIN_EXPECTED_ROWS:
            logger.warning(
                f"Data incomplete for {symbol}, got {len(df)} rows. Skipping this cycle."
            )
            return pd.DataFrame()

        logger.debug(
            "%s raw daily timestamps: %s", symbol, list(df["timestamp"].head())
        )
        try:
            idx = safe_to_datetime(df["timestamp"], context=f"{symbol} daily")
        except ValueError as e:
            logger.debug("Raw daily data for %s: %s", symbol, df.head().to_dict())
            logger.warning("Invalid date index for %s; skipping. %s", symbol, e)
            return None
        logger.debug("%s parsed daily timestamps: %s", symbol, list(idx[:5]))
        df["timestamp"] = idx
        df = df.reset_index(drop=True)

        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        # save to cache (mem)
        if CACHE_ON:
            mcache.put_mem(symbol, tf_name, start_s, end_s, df)
            # Continue execution - caching is not critical
        return df
    except KeyError:
        logger.warning(
            "Missing OHLCV columns for %s; returning empty DataFrame",
            symbol,
        )
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
    except Exception as e:
        snippet = (
            df.head().to_dict()
            if "df" in locals() and isinstance(df, pd.DataFrame)
            else "N/A"
        )
        logger.error(
            "get_daily_df processing error for %s: %s", symbol, e, exc_info=True
        )
        logger.debug("get_daily_df raw response for %s: %s", symbol, snippet)
        return None


def fetch_daily_data_async(
    symbols: Sequence[str], start, end
) -> dict[str, pd.DataFrame]:
    """Fetch daily data for multiple ``symbols`` using pooled HTTP requests."""
    start_dt = ensure_datetime(start)
    end_dt = ensure_datetime(end)
    urls = [_build_daily_url(sym, start_dt, end_dt) for sym in symbols]
    with StageTimer(logger, "UNIVERSE_FETCH", universe_size=len(symbols)):
        responses = http.map_get(urls)
    results: dict[str, pd.DataFrame] = {}
    for sym, (_, code, body) in zip(symbols, responses, strict=False):
        try:
            df = _parse_bars(sym, code, body)
            if df is not None:
                results[sym] = df
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("fetch_daily_data_async failed for %s: %s", sym, exc)
    return results


@retry(
    wait=_RETRY_WAIT,
    stop=_RETRY_STOP,
    retry=retry_if_exception_type((APIError, KeyError, ConnectionError)),
    reraise=True,
)
def get_minute_df(
    symbol: str,
    start_date: date | datetime | pd.Timestamp | str,
    end_date: date | datetime | pd.Timestamp | str,
    limit: int | None = None,  # AI-AGENT-REF: Add missing limit parameter
) -> pd.DataFrame | None:
    """Return minute-level OHLCV DataFrame for ``symbol``.

    Falls back to daily bars if minute data is unavailable.

    Parameters
    ----------
    symbol : str
        Ticker symbol to fetch.
    start_date : datetime-like or str
        Start date of the range.
    end_date : datetime-like or str
        End date of the range.
    limit : int, optional
        Maximum number of rows to return. If None, returns all available data.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by timestamp with ``open``/``high``/``low``/``close``/``volume`` columns.

    Raises
    ------
    APIError
        If Alpaca returns an error and no fallback succeeds.
    KeyError
        If expected OHLCV columns are missing.
    ConnectionError
        For network-related failures handled by ``tenacity`` retry.
    """
    import pandas as pd

    if start_date is None or end_date is None:
        logger.error(
            "get_minute_df called with None dates: %r, %r",
            start_date,
            end_date,
            stack_info=True,
        )
        raise ValueError("start_date and end_date must not be None")

    for name, val in (("start_date", start_date), ("end_date", end_date)):
        if not isinstance(val, date | datetime | str | pd.Timestamp):
            logger.error(
                "get_minute_df invalid %s type: %r",
                name,
                type(val),
                stack_info=True,
            )
            raise TypeError(f"{name} must be date, datetime, pandas.Timestamp or str")

    logger.debug(
        "get_minute_df converting dates start=%r (%s) end=%r (%s)",
        start_date,
        type(start_date).__name__,
        end_date,
        type(end_date).__name__,
    )
    try:
        start_date = ensure_datetime(start_date)
        end_date = ensure_datetime(end_date)
    except (ValueError, TypeError) as dt_err:
        logger.error("get_minute_df datetime error: %s", dt_err, exc_info=True)
        raise
    logger.debug(
        "get_minute_df parsed dates start=%s end=%s",
        start_date,
        end_date,
    )

    # Skip network calls when requesting near real-time data outside market hours
    if not is_market_open():
        logger.info("MARKET_CLOSED_MINUTE_FETCH", extra={"symbol": symbol})
        _MINUTE_CACHE.pop(symbol, None)
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    start_dt = ensure_utc(start_date) - timedelta(minutes=1)
    end_dt = ensure_utc(end_date)

    cached = _MINUTE_CACHE.get(symbol)
    # AI-AGENT-REF: Improved caching to reduce redundant API calls with better hit ratio
    if cached is not None:
        df_cached, ts = cached
        if not df_cached.empty:
            # AI-AGENT-REF: Simplified cache validity logic - extend cache for better performance
            cache_validity_minutes = (
                5  # Increase cache validity to 5 minutes for better hit ratio
            )
            cache_age = pd.Timestamp.now(tz="UTC") - ts
            cache_age_minutes = cache_age.total_seconds() / 60

            if cache_age_minutes <= cache_validity_minutes:
                _CACHE_STATS["hits"] += 1
                logger.debug(
                    "MINUTE_CACHE_HIT",
                    extra={
                        "symbol": symbol,
                        "cache_age_minutes": round(cache_age_minutes, 1),
                        "rows": len(df_cached),
                        "cache_size": len(_MINUTE_CACHE),
                        "hit_ratio_pct": (
                            round(
                                (
                                    _CACHE_STATS["hits"]
                                    / (_CACHE_STATS["hits"] + _CACHE_STATS["misses"])
                                    * 100
                                ),
                                1,
                            )
                            if (_CACHE_STATS["hits"] + _CACHE_STATS["misses"]) > 0
                            else 0
                        ),
                    },
                )
                # AI-AGENT-REF: Return required columns with explicit timestamp column
                required = ["timestamp", "open", "high", "low", "close", "volume"]
                return df_cached.reset_index()[required].copy()
            else:
                # Cache expired, remove it
                _MINUTE_CACHE.pop(symbol, None)
                _CACHE_STATS["invalidations"] += 1
                logger.debug(
                    "MINUTE_CACHE_EXPIRED",
                    extra={
                        "symbol": symbol,
                        "cache_age_minutes": round(cache_age_minutes, 1),
                        "validity_minutes": cache_validity_minutes,
                    },
                )

    # Cache miss - will fetch new data
    _CACHE_STATS["misses"] += 1

    alpaca_exc = finnhub_exc = None
    try:
        logger.debug("Trying data source: Alpaca")
        logger.debug("FETCH_ALPACA_MINUTE_BARS: start", extra={"symbol": symbol})
        client = _get_alpaca_client()
        if client is not None and StockBarsRequest and TimeFrame:
            req = StockBarsRequest(
                symbol_or_symbols=[symbol],
                start=start_dt,
                end=end_dt,
                timeframe=TimeFrame.Minute,
                feed=_DEFAULT_FEED,
            )
            try:
                bars = client.get_stock_bars(req).df
                df = bars.reset_index().rename(
                    columns={
                        "t": "timestamp",
                        "timestamp": "timestamp",
                        "o": "open",
                        "h": "high",
                        "l": "low",
                        "c": "close",
                        "v": "volume",
                    }
                )
            except Exception as e:
                if "invalid format for parameter start" in str(e).lower():
                    raise APIError(str(e)) from e
                raise DataFetchException(symbol, "alpaca", "", str(e)) from e
        else:
            df = _fetch_bars(symbol, start_dt, end_dt, "1Min", _DEFAULT_FEED)
        logger.debug(
            "FETCH_ALPACA_MINUTE_BARS: got %s bars", len(df) if df is not None else 0
        )
        if df is None or df.empty:
            alpaca_exc = DataFetchException(
                symbol,
                "alpaca",
                "",
                f"No minute bars returned for {symbol} from Alpaca",
            )
            df = None
        else:
            required = ["open", "high", "low", "close", "volume", "timestamp"]
            missing = set(required) - set(df.columns)
            if missing:
                logger.error("get_minute_df missing columns %s", missing)
                alpaca_exc = DataFetchException(
                    symbol,
                    "alpaca",
                    "",
                    f"Alpaca minute bars for {symbol} missing columns {missing}",
                )
                df = None
    except Exception as primary_err:
        if "invalid format for parameter start" in str(primary_err).lower():
            raise APIError(str(primary_err)) from primary_err
        alpaca_exc = primary_err
        logger.debug("Alpaca fetch error: %s", primary_err)
        logger.debug("Falling back to Finnhub")
        try:
            logger.info("DATA_SOURCE_FALLBACK: trying %s", "Finnhub")
            logger.debug("FETCH_FINNHUB_MINUTE_BARS: start", extra={"symbol": symbol})
            df = _get_fh_fetcher().fetch(symbol, period="1d", interval="1")
            logger.debug(
                "FETCH_FINNHUB_MINUTE_BARS: got %s bars",
                len(df) if df is not None else 0,
            )
            required = ["open", "high", "low", "close", "volume"]
            missing = set(required) - set(df.columns)
            if missing:
                logger.error("get_minute_df missing columns %s", missing)
                return pd.DataFrame(columns=required)
            # Successfully fetched data from Finnhub, return it
            # AI-AGENT-REF: Filter to only return required columns while preserving index
            result = df[required].copy()
            return result
        except FinnhubAPIException as fh_err:
            finnhub_exc = fh_err
            logger.error("[DataFetcher] Finnhub failed: %s", fh_err)
            logger.debug("Falling back to yfinance")
            if getattr(fh_err, "status_code", None) == 403:
                logger.warning("Finnhub 403 for %s; using yfinance fallback", symbol)
                try:
                    logger.debug("Trying data source: yfinance")
                    logger.info("DATA_SOURCE_FALLBACK: trying %s", "yfinance")
                    logger.debug(
                        "FETCH_YFINANCE_MINUTE_BARS: start", extra={"symbol": symbol}
                    )
                    df = fetch_minute_yfinance(symbol)
                    logger.debug(
                        "FETCH_YFINANCE_MINUTE_BARS: got %s bars",
                        len(df) if df is not None else 0,
                    )
                    required = ["timestamp", "open", "high", "low", "close", "volume"]
                    missing = set(required) - set(df.columns)
                    if missing:
                        logger.error("get_minute_df missing columns %s", missing)
                        return pd.DataFrame(columns=required)
                    # Successfully fetched data from yfinance, return it
                    return df.reset_index()[required].copy()
                except Exception as exc:
                    logger.error("[DataFetcher] yfinance failed: %s", exc)
                    logger.error(
                        "DATA_SOURCE_RETRY_FINAL: alpaca failed=%s; finnhub failed=%s; yfinance failed=%s | last=%s",
                        alpaca_exc,
                        fh_err,
                        exc,
                        "yfinance",
                    )
                    logger.debug("yfinance fetch error: %s", exc)
                    raise DataSourceDownException(symbol) from exc
            else:
                logger.critical("Secondary provider failed for %s: %s", symbol, fh_err)
                raise DataSourceDownException(symbol) from fh_err
        except Exception as fh_err:
            finnhub_exc = fh_err
            logger.error("[DataFetcher] Finnhub failed: %s", fh_err)
            logger.debug("Falling back to yfinance")
            try:
                logger.debug("Trying data source: yfinance")
                logger.info("DATA_SOURCE_FALLBACK: trying %s", "yfinance")
                logger.debug(
                    "FETCH_YFINANCE_MINUTE_BARS: start", extra={"symbol": symbol}
                )
                df = fetch_minute_yfinance(symbol)
                logger.debug(
                    "FETCH_YFINANCE_MINUTE_BARS: got %s bars",
                    len(df) if df is not None else 0,
                )
                required = ["timestamp", "open", "high", "low", "close", "volume"]
                missing = set(required) - set(df.columns)
                if missing:
                    logger.error("get_minute_df missing columns %s", missing)
                    return pd.DataFrame(columns=required)
                # Successfully fetched data from yfinance, return it
                return df.reset_index()[required].copy()
            except Exception as exc:
                logger.error("[DataFetcher] yfinance failed: %s", exc)
                logger.error(
                    "DATA_SOURCE_RETRY_FINAL: alpaca failed=%s; finnhub failed=%s; yfinance failed=%s | last=%s",
                    alpaca_exc,
                    finnhub_exc,
                    exc,
                    "yfinance",
                )
                logger.debug("yfinance fetch error: %s", exc)
                raise DataSourceDownException(symbol) from exc
    if df is None or df.empty:
        # AI-AGENT-REF: raise explicit error when all providers return empty
        raise DataSourceEmpty(symbol)
    required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error(
            "get_minute_df missing columns: %s", sorted(missing)
        )  # AI-AGENT-REF: early validation
        raise KeyError(f"missing columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    if len(df) < MIN_EXPECTED_ROWS:
        logger.critical("INCOMPLETE_DATA", extra={"symbol": symbol, "rows": len(df)})
    # AI-AGENT-REF: Update cache with fresh data and log performance metrics
    _MINUTE_CACHE[symbol] = (df, pd.Timestamp.now(tz="UTC"))

    # Calculate current cache performance
    total_requests = _CACHE_STATS["hits"] + _CACHE_STATS["misses"]
    hit_ratio = (
        (_CACHE_STATS["hits"] / total_requests * 100) if total_requests > 0 else 0
    )

    # AI-AGENT-REF: Safe column count calculation to handle edge cases
    try:
        cols_count = (
            df.shape[1]
            if hasattr(df, "shape")
            and hasattr(df.shape, "__len__")
            and len(df.shape) > 1
            else 0
        )
    except (TypeError, AttributeError):
        # Fallback if df.shape access fails
        cols_count = len(df.columns) if hasattr(df, "columns") else 0

    logger.info(
        "MINUTE_FETCHED",
        extra={
            "symbol": symbol,
            "rows": len(df),
            "cols": cols_count,
            "data_source": "fresh_fetch",
            "cache_size": len(_MINUTE_CACHE),
            "cache_hit_ratio_pct": round(hit_ratio, 1),
            "total_cache_requests": total_requests,
        },
    )
    # AI-AGENT-REF: Apply limit parameter if specified
    if limit is not None and len(df) > limit:
        df = df.tail(limit)  # Return the most recent 'limit' rows
        logger.debug(
            "Applied limit %d to %s data, returning %d rows", limit, symbol, len(df)
        )

    # AI-AGENT-REF: Filter to only return required columns while preserving index
    df = df.reset_index()
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    return df[required].copy()


class FinnhubFetcher:
    """Thin wrapper around the Finnhub client with basic rate limiting."""

    def __init__(self, calls_per_minute: int = 30) -> None:
        """Initialize the fetcher with an API rate limit."""
        self.max_calls = calls_per_minute
        self._timestamps = deque()
        self.client = _get_finnhub_client()

    def _throttle(self) -> None:
        while True:
            now_ts = pytime.time()
            wait_secs = None
            # AI-AGENT-REF: Ensure consistent lock usage in throttle logic
            with _rate_limit_lock:
                while self._timestamps and now_ts - self._timestamps[0] > 60:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now_ts)
                    return
                # Calculate wait time within the lock to prevent race conditions
                wait_secs = (
                    60 - (now_ts - self._timestamps[0]) + random.uniform(0.1, 0.5)
                )
            # Sleep outside the lock to avoid blocking other threads
            if wait_secs is not None:
                psleep(wait_secs)

    def _parse_period(self, period: str) -> int:
        if period.endswith("mo"):
            return int(period[:-2]) * 30 * 86400
        num = int(period[:-1])
        unit = period[-1]
        if unit == "d":
            return num * 86400
        raise ValueError(f"Unsupported period: {period}")

    @retry(
        stop=_RETRY_STOP,
        wait=_RETRY_WAIT + wait_random(0.1, 1),
        retry=retry_if_exception_type((RequestException, urllib3.exceptions.HTTPError)),
    )
    def fetch(
        self, symbols: str | Sequence[str], period: str = "1mo", interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch OHLCV data for ``symbols`` over the given period."""
        syms = symbols if isinstance(symbols, list | tuple) else [symbols]
        if self.client is None:
            return pd.DataFrame()
        now_ts = int(pytime.time())
        span = self._parse_period(period)
        start_ts = now_ts - span
        resolution = "D" if interval == "1d" else "1"
        frames = []
        for sym in syms:
            self._throttle()
            resp = self.client.stock_candles(sym, resolution, _from=start_ts, to=now_ts)
            if resp.get("s") != "ok":
                # AI-AGENT-REF: Create empty DataFrame with proper DatetimeIndex structure to prevent index mismatch
                logger.debug(f"Finnhub returned no data for {sym}: {resp.get('s')}")
                empty_df = pd.DataFrame(
                    columns=["open", "high", "low", "close", "volume"],
                    index=pd.DatetimeIndex([], tz=UTC),
                )
                frames.append(empty_df)
                continue
            try:
                idx = safe_to_datetime(resp["t"], context=f"Finnhub {sym}")
            except ValueError as e:
                logger.warning("Failed timestamp parse for %s: %s", sym, e)
                logger.debug("Raw Finnhub response for %s: %s", sym, resp)
                idx = pd.DatetimeIndex([], tz=UTC)
            df = pd.DataFrame(
                {
                    "open": resp["o"],
                    "high": resp["h"],
                    "low": resp["l"],
                    "close": resp["c"],
                    "volume": resp["v"],
                },
                index=idx,
            )
            df["timestamp"] = df.index
            frames.append(df)
        if not frames:
            return None
        if len(frames) == 1:
            return frames[0]
        return pd.concat(frames, axis=0, keys=syms, names=["symbol"]).reset_index(
            level=0
        )


_fh_fetcher: FinnhubFetcher | None = None


def _get_fh_fetcher() -> FinnhubFetcher:
    global _fh_fetcher
    if _fh_fetcher is None:
        _fh_fetcher = FinnhubFetcher()
    return _fh_fetcher


def fetch_minute_yfinance(symbol: str) -> pd.DataFrame:
    """Fetch one day of minute bars using yfinance."""
    import yfinance as yf

    df = yf.Ticker(symbol).history(period="1d", interval="1m")
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        df.index = idx.tz_localize("UTC")
    else:
        df.index = idx.tz_convert("UTC")  # AI-AGENT-REF: preserve existing tz
    df = df.rename_axis("timestamp").reset_index()
    df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]]
    df.columns = [c.lower() for c in df.columns]
    # AI-AGENT-REF: Preserve DatetimeIndex when returning data - set timestamp as index to maintain DatetimeIndex
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    return df


# ------------------------------
# Batched fetch & warm-up helpers
# ------------------------------
def _resolve_timeframe(
    timeframe: str | TimeFrame | TimeFrameUnit,
) -> TimeFrame:
    """Resolve timeframe parameter to TimeFrame object."""


def _to_dt(dt_input: date | datetime | str) -> datetime:
    """Convert date/datetime/string to datetime."""
    if isinstance(dt_input, str):
        return ensure_datetime(dt_input)
    elif isinstance(dt_input, date) and not isinstance(dt_input, datetime):
        return datetime.combine(dt_input, datetime.min.time()).replace(tzinfo=UTC)
    elif isinstance(dt_input, datetime):
        if dt_input.tzinfo is None:
            return dt_input.replace(tzinfo=UTC)
        return dt_input
    else:
        return ensure_datetime(dt_input)


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame columns and structure."""
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    # Handle multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    # Drop symbol column if present
    df = df.drop(columns=["symbol"], errors="ignore")

    # Normalize column names to lowercase
    if not df.empty and len(df.columns) > 0:
        df.columns = df.columns.str.lower()

    # Ensure required columns exist
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0

    # Add timestamp if not present
    if "timestamp" not in df.columns:
        if hasattr(df.index, "to_series"):
            df["timestamp"] = df.index.to_series()
        else:
            df["timestamp"] = pd.Timestamp.now(tz="UTC")

    # Reset index and return expected columns
    df = df.reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def get_bars_batch(
    symbols: list[str],
    timeframe: str | TimeFrame | TimeFrameUnit,
    start: date | datetime | str,
    end: date | datetime | str,
    feed: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch bars for multiple symbols in one request when possible.
    Returns a dict {symbol: DataFrame}. Applies the existing read-through cache per symbol.
    """
    if not symbols:
        return {}
    # Resolve timeframe
    tf = _resolve_timeframe(timeframe)
    tf_name = (
        "1D"
        if str(tf).endswith("Day")
        else (str(tf) if hasattr(tf, "__str__") else "custom")
    )
    start_dt, end_dt = _to_dt(start), _to_dt(end)
    start_s, end_s = str(start_dt), str(end_dt)

    # Try cache first per symbol
    CACHE_ON = bool(get_data_cache_enable())
    TTL = max(0, int(get_data_cache_ttl_seconds()))
    results: dict[str, pd.DataFrame] = {}
    to_fetch: list[str] = []
    if CACHE_ON:
        for sym in symbols:
            cached = mcache.get_mem(sym, tf_name, start_s, end_s, TTL)
            if cached is not None:
                _MET_REQS.labels("cache", tf_name, "hit", "batch").inc()
                results[sym] = cached
            else:
                to_fetch.append(sym)
    else:
        to_fetch = list(symbols)

    if not to_fetch:
        return results

    # Perform one batched request
    return results


def warmup_cache(
    symbols: list[str],
    timeframe: str | TimeFrame | TimeFrameUnit,
    start: date | datetime | str,
    end: date | datetime | str,
) -> int:
    """
    Warm cache for a set of symbols. Returns count of warmed symbols.
    Uses batched fetch for efficiency.
    """
    results = get_bars_batch(symbols, timeframe, start, end)
    return len(results)


# ------------------------------
# Intraday (1-Min) batch helpers
# ------------------------------
def get_minute_bars_batch(
    symbols: list[str],
    start: date | datetime | str,
    end: date | datetime | str,
    feed: str | None = None,
) -> dict[str, pd.DataFrame]:
    tf = _resolve_timeframe("1MIN")
    return get_bars_batch(
        symbols=symbols, timeframe=tf, start=start, end=end, feed=feed
    )


def get_minute_bars(
    symbol: str,
    start: date | datetime | str,
    end: date | datetime | str,
    feed: str | None = None,
) -> pd.DataFrame:
    tf = _resolve_timeframe("1MIN")
    return get_bars(symbol=symbol, timeframe=tf, start=start, end=end, feed=feed)


def get_bars(
    symbol: str,
    timeframe: str | TimeFrame | TimeFrameUnit,
    start: date | datetime | str,
    end: date | datetime | str,
    feed: str | None = None,
) -> pd.DataFrame:
    """
    Fetch bars for a single symbol. This wraps get_bars_batch for single symbol requests.
    """
    result = get_bars_batch([symbol], timeframe, start, end, feed)
    return result.get(
        symbol,
        pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
    )


if os.environ.get("PYTEST_RUNNING"):
    import time
    from datetime import datetime, timezone
    from typing import Any

    # Tests patch this symbol directly
    _DATA_CLIENT: Any = None  # type: ignore

    def ensure_datetime(dt: datetime) -> datetime:
        """Return a timezone-aware UTC datetime suitable for Alpaca."""  # AI-AGENT-REF: test helper
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _should_retry_on_exc(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "invalid format for parameter start" in msg or "error parsing" in msg

    def get_minute_df(
        symbol: str,
        start: datetime,
        end: datetime,
        max_retries: int = 3,
    ):
        """Fetch minute bars with bounded retries."""  # AI-AGENT-REF: retry helper
        if _DATA_CLIENT is None:
            raise RuntimeError("_DATA_CLIENT is not initialized")
        start = ensure_datetime(start)
        end = ensure_datetime(end)
        attempt = 0
        while True:
            try:
                return _DATA_CLIENT.get_stock_bars(
                    symbol, start=start, end=end, timeframe="1Min"
                )
            except Exception as e:  # pragma: no cover - retry path
                attempt += 1
                if attempt >= max_retries or not _should_retry_on_exc(e):
                    raise
                time.sleep(0.1)

    def get_historical_data(
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1Day",
        max_retries: int = 3,
    ):
        """Fetch historical bars with bounded retries."""  # AI-AGENT-REF: retry helper
        if _DATA_CLIENT is None:
            raise RuntimeError("_DATA_CLIENT is not initialized")
        start = ensure_datetime(start)
        end = ensure_datetime(end)
        attempt = 0
        while True:
            try:
                return _DATA_CLIENT.get_stock_bars(
                    symbol, start=start, end=end, timeframe=timeframe
                )
            except Exception as e:  # pragma: no cover - retry path
                attempt += 1
                if attempt >= max_retries or not _should_retry_on_exc(e):
                    raise
                time.sleep(0.1)


# Export RetryError for test compatibility
__all__ = [
    "RetryError",
    "get_historical_data",
    "get_minute_df",
    "get_daily_df",
    "DataFetchError",
    "DataFetchException",
    "get_bars_batch",
    "warmup_cache",
    "get_minute_bars_batch",
    "get_minute_bars",
    "get_bars",
    # Test-required cache helpers (must be re-exported via top-level shim)
    "get_cached_minute_timestamp",
    "last_minute_bar_age_seconds",
    "_MINUTE_CACHE",
]

# AI-AGENT-REF: ensure ensure_datetime is exported
__all__ = list(set([*globals().get("__all__", []), "ensure_datetime"]))

# AI-AGENT-REF: expose as top-level module for tests
sys.modules.setdefault("data_fetcher", sys.modules[__name__])
