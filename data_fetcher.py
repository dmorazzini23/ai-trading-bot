import random
import time as pytime
from collections import deque
from typing import Sequence
from datetime import date, datetime, timedelta, timezone
import warnings

import threading
import config
import logging

FINNHUB_API_KEY = config.FINNHUB_API_KEY
ALPACA_API_KEY = config.ALPACA_API_KEY
ALPACA_SECRET_KEY = config.ALPACA_SECRET_KEY
ALPACA_BASE_URL = config.ALPACA_BASE_URL

from alpaca.data.historical import StockHistoricalDataClient

# Global Alpaca data client using config credentials
client = StockHistoricalDataClient(
    api_key=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
)

logger = logging.getLogger(__name__)
_rate_limit_lock = threading.Lock()
try:
    import requests
    import urllib3
except Exception as e:  # pragma: no cover - allow missing in test env
    logger.warning("Optional dependencies missing: %s", e)
    import types

    requests = types.SimpleNamespace(
        get=lambda *a, **k: None,
        exceptions=types.SimpleNamespace(RequestException=Exception),
    )
    urllib3 = types.SimpleNamespace(
        exceptions=types.SimpleNamespace(HTTPError=Exception)
    )
from utils import ensure_utc, safe_to_datetime, is_market_open

MINUTES_REQUIRED = 31
HISTORICAL_START = "2025-06-01"
HISTORICAL_END = "2025-06-06"
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.common.exceptions import APIError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    retry_if_exception_type,
    RetryError,
)
import finnhub

# Refresh environment in case this module is executed as a script
config.reload_env()

# In-memory minute bar cache to avoid unnecessary API calls

# In-memory minute bar cache to avoid unnecessary API calls
_MINUTE_CACHE: dict[str, tuple[pd.DataFrame, pd.Timestamp]] = {}

# Helper to coerce dates into datetimes
def ensure_datetime(dt: date | datetime) -> datetime:
    """Return ``datetime`` object for ``dt`` without timezone."""
    if isinstance(dt, datetime):
        return dt
    if isinstance(dt, date):
        return datetime.combine(dt, datetime.min.time())
    raise TypeError(f"Unsupported type for ensure_datetime: {type(dt)!r}")

# Alpaca historical data client
_DATA_CLIENT = client


class DataFetchError(Exception):
    pass


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def get_historical_data(
    symbol: str, start_date, end_date, timeframe: str
) -> pd.DataFrame:
    """Fetch historical bars from Alpaca and ensure OHLCV float columns."""

    start_dt = pd.to_datetime(ensure_utc(ensure_datetime(start_date)))
    end_dt = pd.to_datetime(ensure_utc(ensure_datetime(end_date)))

    tf_map = {
        "1Min": TimeFrame.Minute,
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "1Hour": TimeFrame.Hour,
        "1Day": TimeFrame.Day,
    }
    tf = tf_map.get(timeframe)
    if tf is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    def _fetch(feed: str):
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
        # default to IEX-delayed data to avoid SIP subscription errors
        bars = _fetch("iex")
    except APIError as e:
        if "subscription does not permit querying recent sip data" in str(e).lower():
            try:
                bars = _fetch("iex")
            except APIError as iex_err:
                raise DataFetchError(
                    f"IEX fallback failed for {symbol}: {iex_err}"
                ) from iex_err
        else:
            raise
    except RetryError as e:
        raise DataFetchError(f"Historical fetch failed for {symbol}: {e}") from e

    df = pd.DataFrame(bars)
    if df.empty:
        logger.warning(f"No bar data returned for {symbol}")
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    df = df.drop(columns=["symbol"], errors="ignore")

    df.columns = df.columns.str.lower()

    if not df.empty:
        idx = safe_to_datetime(df.index)
        if idx is None:
            logger.warning(f"Unexpected index format for {symbol}; skipping")
            return pd.DataFrame()
        df.index = idx
        df["timestamp"] = df.index

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise KeyError(f"Missing '{col}' column for {symbol}")
        df[col] = df[col].astype(float)

    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def get_daily_df(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Fetch daily bars with retry and IEX fallback."""
    df = pd.DataFrame()
    for attempt in range(3):
        try:
            df = get_historical_data(
                symbol,
                ensure_datetime(start),
                ensure_datetime(end),
                "1Day",
            )
            break
        except (APIError, RetryError) as e:
            logger.debug(f"get_daily_df attempt {attempt+1} failed for {symbol}: {e}")
            pytime.sleep(1)
    else:
        try:
            req = StockBarsRequest(
                symbol_or_symbols=[symbol],
                start=ensure_utc(ensure_datetime(start)),
                end=ensure_utc(ensure_datetime(end)),
                timeframe=TimeFrame.Day,
                feed="iex",
            )
            df = _DATA_CLIENT.get_stock_bars(req).df
        except (APIError, RetryError):
            logger.info(f"SKIP_NO_PRICE_DATA | {symbol}")
            return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    df = df.drop(columns=["symbol"], errors="ignore")

    df.columns = df.columns.str.lower()

    if df.empty:
        logger.warning(f"No daily bars returned for {symbol}")
        return pd.DataFrame()

    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.get_level_values(0)
    idx = safe_to_datetime(df.index)
    if idx is None:
        logger.warning(f"Invalid date index for {symbol}; skipping")
        return pd.DataFrame()
    df.index = idx
    df["timestamp"] = df.index

    try:
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
    except KeyError:
        logger.warning(f"Missing OHLCV columns for {symbol}; returning empty DataFrame")
        return pd.DataFrame()


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((APIError, KeyError, ConnectionError)),
)
def get_minute_df(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Return minute-level OHLCV DataFrame for ``symbol``.

    Falls back to daily bars if minute data is unavailable.

    Parameters
    ----------
    symbol : str
        Ticker symbol to fetch.
    start_date : date
        Start date of the range.
    end_date : date
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

    start_date = ensure_datetime(start_date)
    end_date = ensure_datetime(end_date)

    # Skip network calls when requesting near real-time data outside market hours
    end_check = end_date
    if hasattr(end_check, "date"):
        end_check = end_check.date()
    if end_check >= datetime.now(timezone.utc).date() and not is_market_open():
        logger.info("MARKET_CLOSED_MINUTE_FETCH", extra={"symbol": symbol})
        return pd.DataFrame()

    start_dt = ensure_utc(start_date) - timedelta(minutes=1)
    end_dt = ensure_utc(end_date)

    # Serve cached data if still fresh (within 1 minute of last bar)
    cached = _MINUTE_CACHE.get(symbol)
    if cached is not None:
        df_cached, ts = cached
        if not df_cached.empty and ts >= pd.Timestamp.utcnow() - pd.Timedelta(minutes=1):
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
            )
            try:
                bars = client.get_stock_bars(req)
            except Exception as err:
                logger.exception("Minute data fetch failed for %s: %s", symbol, err)
                raise
        except APIError as e:
            if "subscription does not permit" in str(e).lower():
                logger.critical(
                    f"API subscription error for {symbol}: {e}. Your Alpaca account does not have the required data subscription (recent SIP data). Please upgrade your Alpaca data plan or change data source."
                )
                req.feed = "iex"
                try:
                    bars = client.get_stock_bars(req)
                except Exception as iex_err:
                    logger.exception("IEX fallback failed for %s: %s", symbol, iex_err)
                    return pd.DataFrame()
            else:
                logger.error(f"API error for {symbol}: {e}")
                return pd.DataFrame()

        if bars is None or not getattr(bars, "df", pd.DataFrame()).size:
            logger.warning(f"No bar data returned for {symbol}, skipping")
            return pd.DataFrame()

        bars = bars.df
        if bars.empty:
            logger.error(
                f"Data fetch failed for {symbol} on {end_dt.date()} during trading hours! Skipping symbol."
            )
            # Optionally, alert or set error counter here
            return pd.DataFrame()
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
            raise KeyError("Missing OHLCV columns")

        df = df[["open", "high", "low", "close", "volume"]]
        if df.empty:
            logger.warning(f"No bar data after column filtering for {symbol}")
            return pd.DataFrame()
        idx = safe_to_datetime(df.index)
        if idx is None:
            logger.warning(f"Invalid minute index for {symbol}; skipping")
            return pd.DataFrame()
        df.index = idx

        _MINUTE_CACHE[symbol] = (df, pd.Timestamp.utcnow())
        logger.info(
            "MINUTE_FETCHED",
            extra={"symbol": symbol, "rows": len(df), "cols": df.shape[1]},
        )
        return df

    except (APIError, KeyError):
        try:
            req = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Day,
                start=ensure_utc(ensure_datetime(start_date)),
                end=ensure_utc(ensure_datetime(end_date)) + timedelta(days=1),
                feed="iex",
            )
            try:
                bars = client.get_stock_bars(req)
            except Exception as fetch_err:
                logger.exception("Daily fallback fetch failed for %s: %s", symbol, fetch_err)
                raise
            df = bars.df[["open", "high", "low", "close", "volume"]].copy()
            if df.empty:
                logger.warning(f"Daily fallback returned no data for {symbol}")
                return pd.DataFrame()
            idx = safe_to_datetime(df.index)
            if idx is None:
                logger.warning(f"Invalid fallback index for {symbol}; skipping")
                return pd.DataFrame()
            df.index = idx
            logger.info(
                f"Falling back to daily bars for {symbol} ({len(df)} rows)")
            _MINUTE_CACHE[symbol] = (df, pd.Timestamp.utcnow())
            logger.info(
                "MINUTE_FETCHED",
                extra={"symbol": symbol, "rows": len(df), "cols": df.shape[1]},
            )
            return df
        except Exception as daily_err:
            logger.debug(f"{symbol}: daily fallback fetch failed: {daily_err}")
            return pd.DataFrame()


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
                wait_secs = (
                    60 - (now_ts - self._timestamps[0]) + random.uniform(0.1, 0.5)
                )
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
        retry=retry_if_exception_type(
            (requests.exceptions.RequestException, urllib3.exceptions.HTTPError)
        ),
    )
    def fetch(
        self, symbols: str | Sequence[str], period: str = "1mo", interval: str = "1d"
    ) -> pd.DataFrame:
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
            df = pd.DataFrame(
                {
                    "open": resp["o"],
                    "high": resp["h"],
                    "low": resp["l"],
                    "close": resp["c"],
                    "volume": resp["v"],
                },
                index=pd.to_datetime(resp["t"], unit="s", utc=True),
            )
            df.index = df.index.tz_convert(None)
            df["timestamp"] = df.index
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        if len(frames) == 1:
            return frames[0]
        return pd.concat(frames, axis=0, keys=syms, names=["symbol"]).reset_index(
            level=0
        )
