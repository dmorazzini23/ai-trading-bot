import random
import time as pytime
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from collections import deque
from typing import Optional, Sequence
import warnings

from dotenv import load_dotenv
from config import (
    FINNHUB_API_KEY,
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_BASE_URL,
)

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
from alpaca_trade_api.rest import APIError
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
def get_historical_data(symbol: str, start_date, end_date, timeframe: str) -> pd.DataFrame:
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
            except Exception as iex_err:
                raise DataFetchError(f"IEX fallback failed for {symbol}: {iex_err}") from iex_err
        else:
            raise
    except Exception as e:
        raise DataFetchError(f"Historical fetch failed for {symbol}: {e}") from e

    df = pd.DataFrame(bars)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(symbol, level=0, axis=1)
    else:
        df = df.drop(columns=["symbol"], errors="ignore")

    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)

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
            req = StockBarsRequest(symbol_or_symbols=[symbol], start=start, end=end, timeframe=TimeFrame.Day, feed="iex")
            df = _DATA_CLIENT.get_stock_bars(req).df
        except Exception:
            logger.info(f"SKIP_NO_PRICE_DATA | {symbol}")
            return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(symbol, level=0, axis=1)
    else:
        df = df.drop(columns=["symbol"], errors="ignore")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df["timestamp"] = df.index
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def get_minute_df(symbol: str, start_date, end_date) -> pd.DataFrame:
    start_dt = pd.to_datetime(start_date).tz_localize("UTC")
    end_dt = pd.to_datetime(end_date).tz_localize("UTC") + pd.Timedelta(days=1)
    df = pd.DataFrame()
    for attempt in range(3):
        try:
            df = get_historical_data(symbol, start_dt, end_dt, "1Min")
            break
        except (APIError, RetryError) as e:
            logger.debug(f"get_minute_df attempt {attempt+1} failed for {symbol}: {e}")
            pytime.sleep(1)
    else:
        try:
            req = StockBarsRequest(symbol_or_symbols=[symbol], start=start_dt, end=end_dt, timeframe=TimeFrame.Minute, feed="iex")
            df = _DATA_CLIENT.get_stock_bars(req).df
        except Exception:
            logger.info(f"SKIP_NO_PRICE_DATA | {symbol}")
            return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(symbol, level=0, axis=1)
    else:
        df = df.drop(columns=["symbol"], errors="ignore")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df["timestamp"] = df.index
    if df.empty or "close" not in df.columns:
        return pd.DataFrame()
    return df[["timestamp", "open", "high", "low", "close", "volume"]]

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

class FinnhubFetcher:
    def __init__(self, calls_per_minute: int = 30) -> None:
        self.max_calls = calls_per_minute
        self._timestamps = deque()
        self.client = finnhub_client

    def _throttle(self) -> None:
        while True:
            now_ts = pytime.time()
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10) + wait_random(0.1, 1), retry=retry_if_exception_type(Exception))
    def fetch(self, symbols, period="1mo", interval="1d") -> pd.DataFrame:
        syms = symbols if isinstance(symbols, (list, tuple)) else [symbols]
        now_ts = int(pytime.time())
        span = self._parse_period(period)
        start_ts = now_ts - span
        resolution = 'D' if interval == '1d' else '1'
        frames = []
        for sym in syms:
            self._throttle()
            resp = self.client.stock_candles(sym, resolution, _from=start_ts, to=now_ts)
            if resp.get("s") != "ok":
                frames.append(pd.DataFrame())
                continue
            df = pd.DataFrame({
                "open": resp["o"],
                "high": resp["h"],
                "low": resp["l"],
                "close": resp["c"],
                "volume": resp["v"],
            }, index=pd.to_datetime(resp["t"], unit="s", utc=True))
            df.index = df.index.tz_convert(None)
            df["timestamp"] = df.index
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        if len(frames) == 1:
            return frames[0]
        return pd.concat(frames, axis=0, keys=syms, names=["symbol"]).reset_index(level=0)

