import logging
import random
import sys
import threading
import time as pytime
import types
import warnings
from collections import deque
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional, Sequence

import requests  # AI-AGENT-REF: ensure requests import for function annotations

# Do not hard fail when running under older Python versions in tests
if sys.version_info < (3, 12, 3):  # pragma: no cover - compat check
    print("Warning: Running under unsupported Python version", file=sys.stderr)

import config

FINNHUB_API_KEY = config.FINNHUB_API_KEY
ALPACA_API_KEY = config.ALPACA_API_KEY
ALPACA_SECRET_KEY = config.ALPACA_SECRET_KEY
ALPACA_BASE_URL = config.ALPACA_BASE_URL
ALPACA_DATA_FEED = config.ALPACA_DATA_FEED
HALT_FLAG_PATH = config.HALT_FLAG_PATH

try:
    from alpaca.data.historical import StockHistoricalDataClient
except Exception:  # pragma: no cover - optional dependency
    StockHistoricalDataClient = object  # type: ignore
    client = None
else:
    # Global Alpaca data client using config credentials
    client = StockHistoricalDataClient(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
    )
logger = logging.getLogger(__name__)

# Default market data feed for Alpaca requests
_DEFAULT_FEED = "iex"


def _mask_headers(headers: dict[str, Any]) -> dict[str, Any]:
    masked = {}
    for k, v in headers.items():
        if isinstance(v, str) and any(x in k.lower() for x in ("key", "token", "secret")):
            masked[k] = config.mask_secret(v)
        else:
            masked[k] = v
    return masked


def _log_http_request(method: str, url: str, params: dict[str, Any], headers: dict[str, Any]) -> None:
    logger.debug("HTTP %s %s params=%s headers=%s", method, url, params, _mask_headers(headers))


def _log_http_response(resp: requests.Response) -> None:
    logger.debug("HTTP_RESPONSE status=%s body=%s", resp.status_code, resp.text[:300])

_rate_limit_lock = threading.Lock()
try:
    import requests
    from requests import Session
    from requests.exceptions import HTTPError, RequestException
    import urllib3
except Exception as e:  # pragma: no cover - allow missing in test env
    logger.warning("Optional dependencies missing: %s", e)
    import types

    requests = types.SimpleNamespace(
        Session=lambda: types.SimpleNamespace(get=lambda *a, **k: None),
        get=lambda *a, **k: None,
        exceptions=types.SimpleNamespace(
            RequestException=Exception,
            HTTPError=Exception,
        ),
    )
    urllib3 = types.SimpleNamespace(exceptions=types.SimpleNamespace(HTTPError=Exception))
from utils import ensure_utc, is_market_open, safe_to_datetime

MINUTES_REQUIRED = 31
MIN_EXPECTED_ROWS = 5
HISTORICAL_START = "2025-06-01"
HISTORICAL_END = "2025-06-06"
import logging

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import finnhub
    from finnhub import FinnhubAPIException
except Exception:  # pragma: no cover - optional dependency
    finnhub = types.SimpleNamespace(Client=lambda *a, **k: None)
    class FinnhubAPIException(Exception):
        """Fallback exception with status code attribute."""

        def __init__(self, *args, status_code=None, **kwargs) -> None:  # AI-AGENT-REF: flex init
            code = status_code if status_code is not None else (args[0] if args else None)
            self.status_code = code
            super().__init__(f"FinnhubAPIException: {code}")
import pandas as pd

try:
    from alpaca.common.exceptions import APIError
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
except Exception:  # pragma: no cover - optional dependency
    APIError = Exception  # type: ignore
    StockBarsRequest = lambda *a, **k: None
    TimeFrame = TimeFrameUnit = types.SimpleNamespace()
from tenacity import (RetryError, retry, retry_if_exception_type,
                      stop_after_attempt, wait_exponential, wait_random)

# Refresh environment in case this module is executed as a script
config.reload_env()

# In-memory minute bar cache to avoid unnecessary API calls

# In-memory minute bar cache to avoid unnecessary API calls
_MINUTE_CACHE: dict[str, tuple[pd.DataFrame, pd.Timestamp]] = {}


def _fetch_bars(symbol: str, start: datetime, end: datetime, timeframe: str, feed: str = _DEFAULT_FEED) -> pd.DataFrame:
    """Fetch raw bars from Alpaca with detailed logging."""
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    params = {
        "start": ensure_utc(start).isoformat(),
        "end": ensure_utc(end).isoformat(),
        "timeframe": timeframe,
        "feed": feed,
    }
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }
    _log_http_request("GET", url, params, headers)
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code == 400 and "invalid feed" in resp.text.lower() and feed != "sip":
            logger.warning("Alpaca invalid feed %s for %s; retrying with SIP", feed, symbol)
            params["feed"] = "sip"
            resp = requests.get(url, params=params, headers=headers, timeout=10)
    except requests.exceptions.RequestException as exc:
        logger.exception("HTTP request error for %s", symbol, exc_info=exc)
        raise DataFetchException(symbol, "alpaca", url, str(exc)) from exc

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
    df.set_index("timestamp", inplace=True)
    return df[["open", "high", "low", "close", "volume"]]


# Helper to coerce dates into datetimes
def ensure_datetime(dt: date | datetime | pd.Timestamp | str | None) -> datetime:
    """Coerce ``dt`` into a ``datetime`` instance.

    Accepts ``datetime`` objects, ``pandas.Timestamp`` objects, ``date`` objects,
    or strings in several supported formats. Strings may be ISO 8601
    (with optional timezone), ``"%Y-%m-%d"``, ``"%Y-%m-%d %H:%M:%S"`` or
    ``"%Y%m%d"``.

    Raises
    ------
    ValueError
        If ``dt`` is ``None``, an empty string, or a string that cannot be
        parsed using the supported formats.
    TypeError
        If ``dt`` is of an unsupported type.
    """

    logger.debug("ensure_datetime called with %r (%s)", dt, type(dt).__name__)

    if dt is None:
        logger.error("ensure_datetime received None", stack_info=True)
        raise ValueError("datetime value cannot be None")

    if dt is pd.NaT or (isinstance(dt, pd.Timestamp) and pd.isna(dt)):
        logger.error("ensure_datetime received NaT", stack_info=True)
        raise ValueError("datetime value cannot be NaT")

    if isinstance(dt, pd.Timestamp):
        logger.debug("ensure_datetime using pandas.Timestamp %r", dt)
        return dt.to_pydatetime()

    if isinstance(dt, datetime):
        logger.debug("ensure_datetime received datetime %r", dt)
        return dt

    if isinstance(dt, date):
        logger.debug("ensure_datetime converting date %r", dt)
        return datetime.combine(dt, datetime.min.time())

    if isinstance(dt, str):
        value = dt.strip()
        if not value:
            logger.error("ensure_datetime received empty string", stack_info=True)
            raise ValueError("datetime string is empty")

        formats = [
            "ISO 8601",
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y%m%d",
        ]

        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            logger.debug("ensure_datetime parsed %r via ISO", value)
            return parsed
        except ValueError:
            pass

        for fmt in formats[1:]:
            try:
                parsed = datetime.strptime(value, fmt)
                logger.debug("ensure_datetime parsed %r with %s", value, fmt)
                return parsed
            except ValueError:
                continue

        logger.error(
            "ensure_datetime failed to parse %r with formats %s",
            value,
            formats,
            stack_info=True,
        )
        raise ValueError(f"Invalid datetime string {value!r}; tried {formats}")

    logger.error(
        "ensure_datetime unsupported type: %r", dt, stack_info=True
    )
    raise TypeError(f"Unsupported type for ensure_datetime: {type(dt)!r}")


# Alpaca historical data client
_DATA_CLIENT = client


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


def get_last_available_bar(symbol: str) -> pd.DataFrame:
    """Return the most recent daily bar for ``symbol`` or empty DataFrame."""
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=2)
    try:
        df = get_daily_df(symbol, start, end)
        if df is not None and not df.empty:
            return df.tail(1)
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("get_last_available_bar failed for %s: %s", symbol, exc)
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])



@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def get_historical_data(
    symbol: str,
    start_date,
    end_date,
    timeframe: str,
    raise_on_empty: bool = False,
) -> pd.DataFrame:
    """Fetch historical bars from Alpaca and ensure OHLCV float columns."""

    if start_date is None or end_date is None:
        logger.error(
            "get_historical_data called with None dates: %r, %r",
            start_date,
            end_date,
            stack_info=True,
        )
        raise ValueError("start_date and end_date must not be None")

    for name, val in (("start_date", start_date), ("end_date", end_date)):
        if not isinstance(val, (date, datetime, str, pd.Timestamp)):
            logger.error(
                "get_historical_data invalid %s type: %r",
                name,
                type(val),
                stack_info=True,
            )
            raise TypeError(
                f"{name} must be date, datetime, pandas.Timestamp or str"
            )

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

    tf_map = {
        "1Min": TimeFrame.Minute,
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "1Hour": TimeFrame.Hour,
        "1Day": TimeFrame.Day,
    }
    tf = tf_map.get(timeframe)
    if tf is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    def _fetch(feed: str = _DEFAULT_FEED):
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            start=start_dt,
            end=end_dt,
            timeframe=tf,
            feed=feed,
        )
        try:
            return _DATA_CLIENT.get_stock_bars(req).df
        except Exception as err:
            logger.exception("Historical data fetch failed for %s: %s", symbol, err)
            raise

    try:
        bars = _fetch(_DEFAULT_FEED)
    except APIError as e:
        if "subscription does not permit" in str(e).lower() and _DEFAULT_FEED != "iex":
            logger.warning("Subscription error for %s with feed %s: %s", symbol, _DEFAULT_FEED, e)
            try:
                bars = _fetch("iex")
            except APIError as iex_err:
                logger.error("IEX fallback failed for %s: %s", symbol, iex_err)
                raise DataFetchError(f"IEX fallback failed for {symbol}: {iex_err}") from iex_err
        else:
            raise
    except RetryError as e:
        raise DataFetchError(f"Historical fetch failed for {symbol}: {e}") from e

    df = pd.DataFrame(bars)
    if df.empty:
        for attempt in range(3):
            pytime.sleep(0.5 * (attempt + 1))
            bars = _fetch(_DEFAULT_FEED)
            df = pd.DataFrame(bars)
            if not df.empty:
                break
        if df.empty or len(df) < MIN_EXPECTED_ROWS:
            logger.warning(
                f"Data incomplete for {symbol}, got {len(df)} rows. Skipping this cycle."
            )
            if raise_on_empty:
                # AI-AGENT-REF: optionally propagate empty-data condition
                raise DataFetchError("DATA_SOURCE_EMPTY")
            return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    df = df.drop(columns=["symbol"], errors="ignore")

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
        df["timestamp"] = df.index

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise KeyError(f"Missing '{col}' column for {symbol}")
        df[col] = df[col].astype(float)

    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def get_daily_df(
    symbol: str,
    start: date | datetime | pd.Timestamp | str,
    end: date | datetime | pd.Timestamp | str,
) -> pd.DataFrame:
    """Fetch daily bars with retry and IEX fallback."""
    if start is None or end is None:
        logger.error(
            "get_daily_df called with None dates: %r, %r",
            start,
            end,
            stack_info=True,
        )
        raise ValueError("start and end must not be None")

    for name, val in (("start", start), ("end", end)):
        if not isinstance(val, (date, datetime, str, pd.Timestamp)):
            logger.error(
                "get_daily_df invalid %s type: %r",
                name,
                type(val),
                stack_info=True,
            )
            raise TypeError(
                f"{name} must be date, datetime, pandas.Timestamp or str"
            )

    start_dt = ensure_datetime(start)
    end_dt = ensure_datetime(end)
    try:
        df = _fetch_bars(symbol, start_dt, end_dt, "1Day", _DEFAULT_FEED)
    except DataFetchException as primary_err:
        logger.error(
            "Primary daily fetch failed for %s: %s", symbol, primary_err.message
        )
        try:
            df = fh_fetcher.fetch(symbol, period="6mo", interval="1d")
        except Exception as fh_err:
            logger.critical(
                "Secondary provider failed for %s: %s", symbol, fh_err
            )
            raise DataSourceDownException(symbol) from fh_err
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

        if isinstance(df.index, pd.MultiIndex):
            df.index = df.index.get_level_values(0)
        logger.debug("%s raw daily timestamps: %s", symbol, list(df.index[:5]))
        try:
            idx = safe_to_datetime(df.index, context=f"{symbol} daily")
        except ValueError as e:
            logger.debug("Raw daily data for %s: %s", symbol, df.head().to_dict())
            logger.warning("Invalid date index for %s; skipping. %s", symbol, e)
            return None
        logger.debug("%s parsed daily timestamps: %s", symbol, list(idx[:5]))
        df.index = idx
        df["timestamp"] = df.index

        return df[["timestamp", "open", "high", "low", "close", "volume"]]
    except KeyError:
        logger.warning(
            "Missing OHLCV columns for %s; returning empty DataFrame",
            symbol,
        )
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    except Exception as e:
        snippet = df.head().to_dict() if "df" in locals() and isinstance(df, pd.DataFrame) else "N/A"
        logger.error("get_daily_df processing error for %s: %s", symbol, e, exc_info=True)
        logger.debug("get_daily_df raw response for %s: %s", symbol, snippet)
        return None


def fetch_daily_data_async(symbols: Sequence[str], start, end) -> dict[str, pd.DataFrame]:
    """Fetch daily data for multiple ``symbols`` concurrently."""
    results: dict[str, pd.DataFrame] = {}

    def worker(sym: str) -> None:
        try:
            df = get_daily_df(sym, start, end)
            if df is not None:
                results[sym] = df
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("fetch_daily_data_async failed for %s: %s", sym, exc)

    threads = [threading.Thread(target=worker, args=(s,)) for s in symbols]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((APIError, KeyError, ConnectionError)),
    reraise=True,
)
def get_minute_df(
    symbol: str,
    start_date: date | datetime | pd.Timestamp | str,
    end_date: date | datetime | pd.Timestamp | str,
) -> Optional[pd.DataFrame]:
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
        if not isinstance(val, (date, datetime, str, pd.Timestamp)):
            logger.error(
                "get_minute_df invalid %s type: %r",
                name,
                type(val),
                stack_info=True,
            )
            raise TypeError(
                f"{name} must be date, datetime, pandas.Timestamp or str"
            )

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
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    start_dt = ensure_utc(start_date) - timedelta(minutes=1)
    end_dt = ensure_utc(end_date)

    cached = _MINUTE_CACHE.get(symbol)
    if cached is not None:
        df_cached, ts = cached
        if not df_cached.empty and ts >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=1):
            logger.debug("minute cache hit for %s", symbol)
            return df_cached.copy()

    alpaca_exc = finnhub_exc = yexc = None
    try:
        logger.debug("Trying data source: Alpaca")
        logger.info("DATA_SOURCE_FALLBACK: trying %s", "Alpaca")
        logger.debug("FETCH_ALPACA_MINUTE_BARS: start", extra={"symbol": symbol})
        df = _fetch_bars(symbol, start_dt, end_dt, "1Min", _DEFAULT_FEED)
        logger.debug(
            "FETCH_ALPACA_MINUTE_BARS: got %s bars", len(df) if df is not None else 0
        )
        required = ["open", "high", "low", "close", "volume"]
        missing = set(required) - set(df.columns)
        if missing:
            logger.error("get_minute_df missing columns %s", missing)
            return pd.DataFrame(columns=required)
    except DataFetchException as primary_err:
        alpaca_exc = primary_err
        logger.debug("Alpaca fetch error: %s", primary_err)
        logger.debug("Falling back to Finnhub")
        try:
            logger.info("DATA_SOURCE_FALLBACK: trying %s", "Finnhub")
            logger.debug(
                "FETCH_FINNHUB_MINUTE_BARS: start", extra={"symbol": symbol}
            )
            df = fh_fetcher.fetch(symbol, period="1d", interval="1")
            logger.debug(
                "FETCH_FINNHUB_MINUTE_BARS: got %s bars", len(df) if df is not None else 0
            )
            required = ["open", "high", "low", "close", "volume"]
            missing = set(required) - set(df.columns)
            if missing:
                logger.error("get_minute_df missing columns %s", missing)
                return pd.DataFrame(columns=required)
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
                        "FETCH_YFINANCE_MINUTE_BARS: got %s bars", len(df) if df is not None else 0
                    )
                    required = ["open", "high", "low", "close", "volume"]
                    missing = set(required) - set(df.columns)
                    if missing:
                        logger.error("get_minute_df missing columns %s", missing)
                        return pd.DataFrame(columns=required)
                except Exception as exc:
                    yexc = exc
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
                    logger.critical(
                        "Secondary provider failed for %s: %s", symbol, fh_err
                    )
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
                    "FETCH_YFINANCE_MINUTE_BARS: got %s bars", len(df) if df is not None else 0
                )
                required = ["open", "high", "low", "close", "volume"]
                missing = set(required) - set(df.columns)
                if missing:
                    logger.error("get_minute_df missing columns %s", missing)
                    return pd.DataFrame(columns=required)
            except Exception as exc:
                yexc = exc
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
        logger.critical(
            "EMPTY_DATA", extra={"symbol": symbol, "start": start_dt.isoformat(), "end": end_dt.isoformat()}
        )
        logger.warning("Minute-data fetch failed; returning empty DataFrame")
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error("get_minute_df missing columns: %s", sorted(missing))  # AI-AGENT-REF: early validation
        raise KeyError(f"missing columns: {sorted(missing)}")
    if len(df) < MIN_EXPECTED_ROWS:
        logger.critical(
            "INCOMPLETE_DATA", extra={"symbol": symbol, "rows": len(df)}
        )
    _MINUTE_CACHE[symbol] = (df, pd.Timestamp.now(tz="UTC"))
    logger.info(
        "MINUTE_FETCHED",
        extra={"symbol": symbol, "rows": len(df), "cols": df.shape[1]},
    )
    return df


finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)


class FinnhubFetcher:
    """Thin wrapper around the Finnhub client with basic rate limiting."""

    def __init__(self, calls_per_minute: int = 30) -> None:
        """Initialize the fetcher with an API rate limit."""
        self.max_calls = calls_per_minute
        self._timestamps = deque()
        self.client = finnhub_client

    def _throttle(self) -> None:
        while True:
            now_ts = pytime.time()
            with _rate_limit_lock:
                while self._timestamps and now_ts - self._timestamps[0] > 60:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now_ts)
                    return
                wait_secs = 60 - (now_ts - self._timestamps[0]) + random.uniform(0.1, 0.5)
            pytime.sleep(wait_secs)

    def _parse_period(self, period: str) -> int:
        if period.endswith("mo"):
            return int(period[:-2]) * 30 * 86400
        num = int(period[:-1])
        unit = period[-1]
        if unit == "d":
            return num * 86400
        raise ValueError(f"Unsupported period: {period}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10) + wait_random(0.1, 1),
        retry=retry_if_exception_type((RequestException, urllib3.exceptions.HTTPError)),
    )
    def fetch(self, symbols: str | Sequence[str], period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
        """Fetch OHLCV data for ``symbols`` over the given period."""
        syms = symbols if isinstance(symbols, (list, tuple)) else [symbols]
        now_ts = int(pytime.time())
        span = self._parse_period(period)
        start_ts = now_ts - span
        resolution = "D" if interval == "1d" else "1"
        frames = []
        for sym in syms:
            self._throttle()
            resp = self.client.stock_candles(sym, resolution, _from=start_ts, to=now_ts)
            if resp.get("s") != "ok":
                frames.append(pd.DataFrame())
                continue
            try:
                idx = safe_to_datetime(resp["t"], context=f"Finnhub {sym}")
            except ValueError as e:
                logger.warning("Failed timestamp parse for %s: %s", sym, e)
                logger.debug("Raw Finnhub response for %s: %s", sym, resp)
                idx = pd.DatetimeIndex([], tz=timezone.utc)
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
        return pd.concat(frames, axis=0, keys=syms, names=["symbol"]).reset_index(level=0)


fh_fetcher = FinnhubFetcher()


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
    df = df.set_index("timestamp")[["Open", "High", "Low", "Close", "Volume"]]
    df.columns = [c.lower() for c in df.columns]
    return df
