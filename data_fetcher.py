import logging
import random
import sys
import threading
import time as pytime
import types
import warnings
from collections import deque
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Sequence

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

_rate_limit_lock = threading.Lock()
try:
    import requests
    from requests import Session, HTTPError
    from requests.exceptions import RequestException
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
HISTORICAL_START = "2025-06-01"
HISTORICAL_END = "2025-06-06"
import logging

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import finnhub
except Exception:  # pragma: no cover - optional dependency
    finnhub = types.SimpleNamespace(Client=lambda *a, **k: None)
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
_DEFAULT_FEED = ALPACA_DATA_FEED or "iex"


class DataFetchError(Exception):
    """Raised when a data request fails after all retries."""



@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def get_historical_data(symbol: str, start_date, end_date, timeframe: str) -> pd.DataFrame:
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
        logger.critical("NO_DATA_RETURNED_%s", symbol)
        return None

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

    df = pd.DataFrame()
    for attempt in range(3):
        try:
            start_dt = ensure_datetime(start)
            end_dt = ensure_datetime(end)
        except (ValueError, TypeError) as dt_err:
            logger.error("get_daily_df datetime error: %s", dt_err, exc_info=True)
            raise

        try:
            df = get_historical_data(
                symbol,
                start_dt,
                end_dt,
                "1Day",
            )
            break
        except (APIError, RetryError) as e:
            logger.debug(
                f"get_daily_df attempt {attempt+1} failed for {symbol}: {e}"
            )
            pytime.sleep(1)
    else:
        try:
            start_dt = ensure_datetime(start)
            end_dt = ensure_datetime(end)
        except (ValueError, TypeError) as dt_err:
            logger.error("get_daily_df datetime error: %s", dt_err, exc_info=True)
            raise

        try:
            req = StockBarsRequest(
                symbol_or_symbols=[symbol],
                start=ensure_utc(start_dt),
                end=ensure_utc(end_dt),
                timeframe=TimeFrame.Day,
                feed=_DEFAULT_FEED,
            )
            try:
                df = _DATA_CLIENT.get_stock_bars(req).df
            except APIError as e:
                if "subscription does not permit" in str(e).lower() and _DEFAULT_FEED != "iex":
                    logger.warning(
                        "Daily fetch subscription error for %s with feed %s: %s",
                        symbol,
                        _DEFAULT_FEED,
                        e,
                    )
                    req.feed = "iex"
                    df = _DATA_CLIENT.get_stock_bars(req).df
                else:
                    raise
        except (APIError, RetryError):
            logger.info("SKIP_NO_PRICE_DATA | %s", symbol)
            return None
    try:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)
        df = df.drop(columns=["symbol"], errors="ignore")

        df.columns = df.columns.str.lower()

        if df.empty:
            logger.critical("NO_DATA_RETURNED_%s", symbol)
            return None

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

    # Serve cached data if still fresh (within 1 minute of last bar)
    cached = _MINUTE_CACHE.get(symbol)
    if cached is not None:
        df_cached, ts = cached
        if not df_cached.empty and ts >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=1):
            logger.debug("minute cache hit for %s", symbol)
            return df_cached.copy()

    try:
        bars = None
        try:
            req = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Minute,
                start=start_dt,
                end=end_dt,
                feed=_DEFAULT_FEED,
            )
            try:
                bars = client.get_stock_bars(req)
            except Exception as err:
                logger.exception("Minute data fetch failed for %s: %s", symbol, err)
                raise
        except APIError as e:
            if "subscription does not permit" in str(e).lower() and _DEFAULT_FEED != "iex":
                logger.error(
                    "API subscription error for %s with feed %s: %s",
                    symbol,
                    _DEFAULT_FEED,
                    e,
                )
                req.feed = "iex"
                try:
                    bars = client.get_stock_bars(req)
                except Exception as iex_err:
                    logger.exception("IEX fallback failed for %s: %s", symbol, iex_err)
                    return None
            else:
                logger.error(f"API error for {symbol}: {e}")
                return None

        if bars is None or not getattr(bars, "df", pd.DataFrame()).size:
            logger.critical("NO_DATA_RETURNED_%s", symbol)
            return None

        bars = bars.df
        logger.debug("%s raw minute timestamps: %s", symbol, list(bars.index[:5]))
        if bars.empty:
            logger.error(f"Data fetch failed for {symbol} on {end_dt.date()} during trading hours! Skipping symbol.")
            # Optionally, alert or set error counter here
            return None
        # drop MultiIndex if present, otherwise drop the stray "symbol" column
        if isinstance(bars.columns, pd.MultiIndex):
            bars = bars.xs(symbol, level=0, axis=1)
        else:
            bars = bars.drop(columns=["symbol"], errors="ignore")
        df = bars.copy()

        logger.debug(f"{symbol}: raw bar columns: {df.columns.tolist()}")

        rename_map = {}
        patterns = {
            "open": ["open", "o", "open_price"],
            "high": ["high", "h", "high_price"],
            "low": ["low", "l", "low_price"],
            "close": ["close", "c", "close_price"],
            "volume": ["volume", "v"],
        }
        for std, pats in patterns.items():
            for pat in pats:
                for col in df.columns:
                    if col.lower().startswith(pat):
                        rename_map[col] = std
                        break
                if std in rename_map:
                    break

        df.rename(columns=rename_map, inplace=True)
        logger.debug(f"{symbol}: renamed bar columns: {df.columns.tolist()}")

        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            logger.warning("Missing OHLCV columns for %s; returning empty", symbol)
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = df[["open", "high", "low", "close", "volume"]]
        if df.empty:
            logger.critical("NO_DATA_RETURNED_%s", symbol)
            return None
        try:
            idx = safe_to_datetime(df.index, context=f"{symbol} minute")
        except ValueError as e:
            logger.debug("Raw minute data for %s: %s", symbol, df.head().to_dict())
            logger.warning("Invalid minute index for %s; skipping. %s", symbol, e)
            return None
        df.index = idx

        _MINUTE_CACHE[symbol] = (df, pd.Timestamp.now(tz="UTC"))
        logger.info(
            "MINUTE_FETCHED",
            extra={"symbol": symbol, "rows": len(df), "cols": df.shape[1]},
        )
        return df
    except (APIError, KeyError):
        try:
            start_dt = ensure_datetime(start_date)
            end_dt = ensure_datetime(end_date)
        except (ValueError, TypeError) as dt_err:
            logger.error(
                "get_minute_df fallback datetime error: %s", dt_err, exc_info=True
            )
            return None

        try:
            req = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Day,
                start=ensure_utc(start_dt),
                end=ensure_utc(end_dt) + timedelta(days=1),
                feed=_DEFAULT_FEED,
            )
            try:
                bars = client.get_stock_bars(req)
            except APIError as e:
                if "subscription does not permit" in str(e).lower() and _DEFAULT_FEED != "iex":
                    logger.warning(
                        "Minute fallback daily subscription error for %s with feed %s: %s",
                        symbol,
                        _DEFAULT_FEED,
                        e,
                    )
                    req.feed = "iex"
                    bars = client.get_stock_bars(req)
                else:
                    raise
            except Exception as fetch_err:
                logger.exception("Daily fallback fetch failed for %s: %s", symbol, fetch_err)
                raise
            df = bars.df[["open", "high", "low", "close", "volume"]].copy()
            if df.empty:
                logger.critical("NO_DATA_RETURNED_%s", symbol)
                return None
            try:
                idx = safe_to_datetime(df.index, context=f"{symbol} fallback")
            except ValueError as e:
                logger.debug("Raw fallback data for %s: %s", symbol, df.head().to_dict())
                logger.warning(
                    "Invalid fallback index for %s; skipping | %s",
                    symbol,
                    e,
                )
                return None
            logger.debug("%s fallback raw timestamps: %s", symbol, list(df.index[:5]))
            logger.debug("%s fallback parsed timestamps: %s", symbol, list(idx[:5]))
            df.index = idx
            logger.info(
                "Falling back to daily bars for %s (%s rows)",
                symbol,
                len(df),
            )
            _MINUTE_CACHE[symbol] = (df, pd.Timestamp.now(tz="UTC"))
            logger.info(
                "MINUTE_FETCHED",
                extra={"symbol": symbol, "rows": len(df), "cols": df.shape[1]},
            )
            return df
        except Exception as daily_err:
            logger.debug(f"{symbol}: daily fallback fetch failed: {daily_err}")
            return None
    except Exception as e:
        snippet = df.head().to_dict() if "df" in locals() and isinstance(df, pd.DataFrame) else "N/A"
        logger.error("get_minute_df processing error for %s: %s", symbol, e, exc_info=True)
        logger.debug("get_minute_df raw response for %s: %s", symbol, snippet)
        return None


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
