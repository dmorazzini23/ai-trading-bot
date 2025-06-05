import random
import time as pytime
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from collections import deque
from typing import Optional, Sequence

from dotenv import load_dotenv
from config import (
    FINNHUB_API_KEY,
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_BASE_URL,
)
import logging

logger = logging.getLogger(__name__)

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.common.exceptions import APIError
from tenacity import retry, stop_after_attempt, wait_exponential, wait_random, retry_if_exception_type
import finnhub

load_dotenv()

_DATA_CLIENT = StockHistoricalDataClient(
    api_key=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
    feed="iex",
)

class DataFetchError(Exception):
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10) + wait_random(0.1, 0.5),
    retry=retry_if_exception_type(Exception),
)
def get_historical_data(symbol: str, start_date: date, end_date: date, timeframe: str) -> pd.DataFrame:
    """Fetch historical bars for a symbol from Alpaca."""
    tf_map = {
        '1Min': TimeFrame.Minute,
        '5Min': TimeFrame(5, TimeFrameUnit.Minute),
        '1Hour': TimeFrame.Hour,
        '1Day': TimeFrame.Day,
    }
    tf = tf_map.get(timeframe)
    if tf is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        start=datetime.combine(start_date, datetime.min.time(), timezone.utc),
        end=datetime.combine(end_date, datetime.max.time(), timezone.utc),
        timeframe=tf,
    )
    try:
        bars = _DATA_CLIENT.get_stock_bars(req).df
    except Exception as e:
        logger.warning(f"[get_historical_data] API error for {symbol}: {e}")
        raise
    if isinstance(bars.columns, pd.MultiIndex):
        bars = bars.xs(symbol, level=0, axis=1)
    else:
        bars = bars.drop(columns=["symbol"], errors="ignore")
    bars.index = pd.to_datetime(bars.index).tz_localize(None)
    return bars.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
    })

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

class FinnhubFetcher:
    def __init__(self, calls_per_minute: int = 60) -> None:
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
            if resp.get('s') != 'ok':
                frames.append(pd.DataFrame())
                continue
            df = pd.DataFrame({
                'Open': resp['o'],
                'High': resp['h'],
                'Low': resp['l'],
                'Close': resp['c'],
                'Volume': resp['v'],
            }, index=pd.to_datetime(resp['t'], unit='s', utc=True))
            df.index = df.index.tz_convert(None)
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        if len(frames) == 1:
            return frames[0]
        return pd.concat(frames, axis=1, keys=syms, names=['Symbol','Field'])

fh = FinnhubFetcher()

@dataclass
class DataFetcher:
    """Unified data adapter supporting multiple feeds."""

    def __post_init__(self) -> None:
        self._daily_cache: dict[str, Optional[pd.DataFrame]] = {}
        self._minute_cache: dict[str, Optional[pd.DataFrame]] = {}
        self._minute_timestamps: dict[str, datetime] = {}

    def get_daily_df(self, ctx, symbol: str) -> Optional[pd.DataFrame]:
        if symbol in self._daily_cache:
            return self._daily_cache[symbol]
        try:
            end = date.today()
            start = end - timedelta(days=365)
            df = get_historical_data(symbol, start, end, '1Day')
        except APIError as e:
            logger.warning(f"[get_daily_df] Alpaca fetch failed for {symbol}: {e}")
            try:
                df = fh.fetch(symbol, period="1mo", interval="1d")
            except Exception:
                df = None
        except Exception as e:
            logger.warning(f"[get_daily_df] Alpaca fetch failed for {symbol}: {e}")
            try:
                df = fh.fetch(symbol, period="1mo", interval="1d")
            except Exception:
                df = None
        self._daily_cache[symbol] = df
        return df

    def get_minute_df(self, ctx, symbol: str) -> Optional[pd.DataFrame]:
        now = datetime.now(timezone.utc)
        ts = self._minute_timestamps.get(symbol)
        if ts and (now - ts) < timedelta(seconds=60):
            return self._minute_cache.get(symbol)
        df = None
        try:
            end = date.today()
            start = end - timedelta(days=5)
            df = get_historical_data(symbol, start, end, '1Min')
            if df is not None and not df.empty:
                df = df[["Open", "High", "Low", "Close", "Volume"]]
        except APIError as e:
            logger.warning(f"[get_minute_df] Alpaca fetch failed for {symbol}: {e}")
            try:
                df = fh.fetch(symbol, period="5d", interval="1m")
            except Exception:
                df = None
        except Exception as e:
            logger.warning(f"[get_minute_df] Alpaca fetch failed for {symbol}: {e}")
            try:
                df = fh.fetch(symbol, period="5d", interval="1m")
            except Exception:
                df = None
        self._minute_cache[symbol] = df
        self._minute_timestamps[symbol] = now
        return df
