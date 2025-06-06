import random
import time as pytime
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from collections import deque
from typing import Optional, Sequence
import warnings

import os
import threading
from dotenv import load_dotenv
import config

FINNHUB_API_KEY = config.FINNHUB_API_KEY
ALPACA_API_KEY = config.ALPACA_API_KEY
ALPACA_SECRET_KEY = config.ALPACA_SECRET_KEY
ALPACA_BASE_URL = config.ALPACA_BASE_URL

import alpaca_trade_api as tradeapi

# Global Alpaca client using config credentials
client = tradeapi.REST(
    key_id=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
    base_url=ALPACA_BASE_URL,
)
# Alias used throughout the project
api = client

_rate_limit_lock = threading.Lock()
import requests
import urllib3

MINUTES_REQUIRED = 31
HISTORICAL_START = "2025-06-01"
HISTORICAL_END = "2025-06-06"
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca_trade_api.rest import REST, APIError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    retry_if_exception_type,
    wait_fixed,
    RetryError,
)
import finnhub

load_dotenv()

# Alpaca historical data client
_DATA_CLIENT = StockHistoricalDataClient(
    api_key=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
)


class DataFetchError(Exception):
    pass


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def get_historical_data(
    symbol: str, start_date, end_date, timeframe: str
) -> pd.DataFrame:
    """Fetch historical bars from Alpaca and ensure OHLCV float columns."""

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

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
        return _DATA_CLIENT.get_stock_bars(req).df

    try:
        bars = _fetch("sip")
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
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    df = df.drop(columns=["symbol"], errors="ignore")

    df.columns = df.columns.str.lower()

    df.index = [ts[0] if isinstance(ts, tuple) else ts for ts in df.index]
    df.index = pd.to_datetime(df.index, errors="coerce").tz_localize(None)

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
            df = get_historical_data(symbol, start, end, "1Day")
            break
        except (APIError, RetryError) as e:
            logger.debug(f"get_daily_df attempt {attempt+1} failed for {symbol}: {e}")
            pytime.sleep(1)
    else:
        try:
            req = StockBarsRequest(
                symbol_or_symbols=[symbol],
                start=start,
                end=end,
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

    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.get_level_values(0)
    df.index = [ts[0] if isinstance(ts, tuple) else ts for ts in df.index]
    df.index = pd.to_datetime(df.index, errors="coerce").tz_localize(None)
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
    from alpaca_trade_api.rest import TimeFrame, APIError
    import pandas as pd

    try:
        bars = None
        try:
            bars = client.get_bars(
                symbol, TimeFrame.Minute, start=start_date, end=end_date
            )
        except APIError as e:
            if "subscription does not permit" in str(e).lower():
                bars = client.get_bars(
                    symbol,
                    TimeFrame.Minute,
                    start=start_date,
                    end=end_date,
                    adjustment="raw",
                    feed="iex",
                )
            else:
                raise

        if bars is None or not getattr(bars, "df", pd.DataFrame()).size:
            raise APIError("No minute data returned")

        df = bars.df.copy()

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
        df.index = pd.to_datetime(df.index).tz_localize(None)

        return df

    except (APIError, KeyError) as e:
        try:
            bars = client.get_bars(
                symbol,
                timeframe="1Day",
                start=start_date.isoformat(),
                end=(end_date + timedelta(days=1)).isoformat(),
                adjustment="raw",
            )
            df = bars.df[["open", "high", "low", "close", "volume"]].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            logger.info(f"Falling back to daily bars for {symbol} ({len(df)} rows)")
            return df
        except Exception as daily_err:
            logger.debug(f"{symbol}: daily fallback fetch failed: {daily_err}")
            return pd.DataFrame()


finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)


class FinnhubFetcher:
    def __init__(self, calls_per_minute: int = 30) -> None:
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
    def fetch(self, symbols, period="1mo", interval="1d") -> pd.DataFrame:
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
