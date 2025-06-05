import os
import random
import time as pytime
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from collections import deque
from typing import Optional, Sequence

import pandas as pd
import yfinance as yf
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.common.exceptions import APIError
from tenacity import retry, stop_after_attempt, wait_exponential, wait_random, retry_if_exception_type
import finnhub

class DataFetchError(Exception):
    pass

finnhub_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))

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
            # alpaca-py requires a list of symbols; handle resulting MultiIndex
            req = StockBarsRequest(symbol_or_symbols=[symbol], timeframe=TimeFrame.Day, limit=1000)
            bars = ctx.data_client.get_stock_bars(req).df
            if isinstance(bars.columns, pd.MultiIndex):
                bars = bars.xs(symbol, level=0, axis=1)
            else:
                bars = bars.drop(columns=["symbol"], errors="ignore")
            bars.index = pd.to_datetime(bars.index).tz_localize(None)
            df = bars.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        except Exception:
            try:
                df = fh.fetch(symbol, period="1mo", interval="1d")
            except Exception:
                df_yf = yf.download(symbol, period="1mo", interval="1d", progress=False)
                if df_yf.empty:
                    df = None
                else:
                    df_yf.index = pd.to_datetime(df_yf.index).tz_localize(None)
                    df = df_yf.rename(columns=lambda c: c.title())[ ["Open","High","Low","Close","Volume"] ]
        self._daily_cache[symbol] = df
        return df

    def get_minute_df(self, ctx, symbol: str) -> Optional[pd.DataFrame]:
        now = datetime.now(timezone.utc)
        ts = self._minute_timestamps.get(symbol)
        if ts and (now - ts) < timedelta(seconds=60):
            return self._minute_cache.get(symbol)
        df = None
        try:
            req = StockBarsRequest(symbol_or_symbols=[symbol], timeframe=TimeFrame.Minute, limit=390)
            bars = ctx.data_client.get_stock_bars(req).df
            if isinstance(bars.columns, pd.MultiIndex):
                bars = bars.xs(symbol, level=0, axis=1)
            else:
                bars = bars.drop(columns=["symbol"], errors="ignore")
            if not bars.empty:
                bars = bars.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
                bars.index = pd.to_datetime(bars.index).tz_localize(None)
                df = bars[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            try:
                df = fh.fetch(symbol, period="5d", interval="1m")
            except Exception:
                df_yf = yf.download(symbol, period="5d", interval="1m", progress=False)
                if not df_yf.empty:
                    df_yf.index = pd.to_datetime(df_yf.index).tz_localize(None)
                    df = df_yf.rename(columns=lambda c: c.title())[ ["Open","High","Low","Close","Volume"] ]
        self._minute_cache[symbol] = df
        self._minute_timestamps[symbol] = now
        return df
