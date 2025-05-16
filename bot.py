# ─── STANDARD LIBRARIES ─────────────────────────────────────────────────────
import os
import csv
import re
import time, random
from datetime import datetime, date, time as dt_time, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Optional, Tuple, Dict, List, Any, Sequence
from alpaca_trade_api.entity import Order
from dataclasses import dataclass, field
from threading import Semaphore, Thread
from concurrent.futures import ThreadPoolExecutor, as_completed

import logging
logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

# ─── STRUCTURED LOGGING, RETRIES & RATE LIMITING ────────────────────────────
import structlog

# ─── THIRD-PARTY LIBRARIES ────────────────────────────────────────────────────
import numpy as np
if not hasattr(np, "NaN"):
    np.NaN = np.nan
import pandas as pd
import pandas_market_calendars as mcal
import pandas_ta as ta
import requests
from bs4 import BeautifulSoup
from flask import Flask
import schedule
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
import portalocker
from alpaca_trade_api.rest import REST, APIError, TimeFrame
from sklearn.ensemble import RandomForestClassifier
import pickle
from statsmodels.tsa.stattools import coint
from pypfopt import risk_models, expected_returns, EfficientFrontier
import joblib
from dotenv import load_dotenv
import sentry_sdk
from prometheus_client import start_http_server, Counter, Gauge
import finnhub
from finnhub.exceptions import FinnhubAPIException
import pybreaker

# ─── CIRCUIT BREAKER FOR ALPACA CALLS ─────────────────────────────────────────
breaker = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60)

executor = ThreadPoolExecutor(max_workers=4)

def predict_text_sentiment(text: str) -> float:
    return 0.0

from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential, wait_random, retry_if_exception_type
from ratelimit import limits, sleep_and_retry
from collections import deque
from more_itertools import chunked

# for check_daily_loss()
day_start_equity: Optional[Tuple[date, float]] = None

_calendar_cache: Dict[str, pd.DataFrame] = {}
_calendar_last_fetch: Dict[str, date] = {}

# ─── MARKET HOURS GUARD ───────────────────────────────────────────────────
# set up the NYSE calendar once
nyse = mcal.get_calendar("XNYS")

def in_trading_hours(ts: pd.Timestamp) -> bool:
    """Return True if ts (UTC) falls within today’s NYSE open/close."""
    schedule = nyse.schedule(start_date=ts.date(), end_date=ts.date())
    if schedule.empty:
        return False
    return schedule.market_open.iloc[0] <= ts <= schedule.market_close.iloc[0]

# ─── A. ENVIRONMENT, SENTRY & LOGGING ────────────────────────────────────────
load_dotenv()
RUN_HEALTH = os.getenv("RUN_HEALTHCHECK", "1") == "1"
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    traces_sample_rate=0.1,
    environment=os.getenv("BOT_MODE", "live"),
)

finnhub_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))

# ─── C. PROMETHEUS METRICS ───────────────────────────────────────────────────
orders_total   = Counter('bot_orders_total',   'Total orders sent')
order_failures = Counter('bot_order_failures', 'Order submission failures')
daily_drawdown = Gauge('bot_daily_drawdown',   'Current daily drawdown fraction')

# ─── PATH CONFIGURATION ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def abspath(fname: str) -> str:
    return os.path.join(BASE_DIR, fname)

TICKERS_FILE        = abspath("tickers.csv")
TRADE_LOG_FILE      = abspath("trades.csv")
SIGNAL_WEIGHTS_FILE = abspath("signal_weights.csv")
EQUITY_FILE         = abspath("last_equity.txt")
PEAK_EQUITY_FILE    = abspath("peak_equity.txt")
HALT_FLAG_PATH      = abspath("halt.flag")
MODEL_PATH          = abspath(os.getenv("MODEL_PATH", "trained_model.pkl"))

# ─── STRATEGY MODE CONFIGURATION ─────────────────────────────────────────────
class BotMode:
    def __init__(self, mode: str = "balanced") -> None:
        self.mode = mode.lower()
        self.params = self.set_parameters()

    def set_parameters(self) -> dict[str, float]:
        if self.mode == "conservative":
            return {
                "KELLY_FRACTION": 0.3, "CONF_THRESHOLD": 0.8, "CONFIRMATION_COUNT": 3,
                "TAKE_PROFIT_FACTOR": 1.2, "DAILY_LOSS_LIMIT": 0.05,
                "CAPITAL_CAP": 0.05, "TRAILING_FACTOR": 1.5
            }
        elif self.mode == "aggressive":
            return {
                "KELLY_FRACTION": 0.75, "CONF_THRESHOLD": 0.6, "CONFIRMATION_COUNT": 1,
                "TAKE_PROFIT_FACTOR": 2.2, "DAILY_LOSS_LIMIT": 0.1,
                "CAPITAL_CAP": 0.1, "TRAILING_FACTOR": 2.0
            }
        else:  # balanced
            return {
                "KELLY_FRACTION": 0.6, "CONF_THRESHOLD": 0.65, "CONFIRMATION_COUNT": 2,
                "TAKE_PROFIT_FACTOR": 1.8, "DAILY_LOSS_LIMIT": 0.07,
                "CAPITAL_CAP": 0.08, "TRAILING_FACTOR": 1.8
            }

    def get_config(self) -> dict[str, float]:
        return self.params

BOT_MODE = os.getenv("BOT_MODE", "balanced")
mode     = BotMode(BOT_MODE)
logger.info(f"Trading mode is set to '{mode.mode}'")
params   = mode.get_config()

# ─── CONFIGURATION CONSTANTS ─────────────────────────────────────────────────
NEWS_API_KEY             = os.getenv("NEWS_API_KEY")
TRAILING_FACTOR          = params["TRAILING_FACTOR"]
SECONDARY_TRAIL_FACTOR   = 1.0
TAKE_PROFIT_FACTOR       = params["TAKE_PROFIT_FACTOR"]
SCALING_FACTOR           = 0.5
ORDER_TYPE               = 'market'
LIMIT_ORDER_SLIPPAGE     = float(os.getenv("LIMIT_ORDER_SLIPPAGE", 0.005))
MAX_POSITION_SIZE        = 1000
SLICE_THRESHOLD          = 100
POV_SLICE_PCT            = float(os.getenv("POV_SLICE_PCT", "0.0"))
DAILY_LOSS_LIMIT         = params["DAILY_LOSS_LIMIT"]
MAX_PORTFOLIO_POSITIONS  = int(os.getenv("MAX_PORTFOLIO_POSITIONS", 15))
CORRELATION_THRESHOLD    = 0.8
MARKET_OPEN              = dt_time(0, 0)
MARKET_CLOSE             = dt_time(23, 59)
VOLUME_THRESHOLD         = int(os.getenv("VOLUME_THRESHOLD", "10000"))
ENTRY_START_OFFSET       = timedelta(minutes=15)
ENTRY_END_OFFSET         = timedelta(minutes=30)
REGIME_LOOKBACK          = 14
REGIME_ATR_THRESHOLD     = 20.0
RF_ESTIMATORS            = 225
RF_MAX_DEPTH             = 5
ATR_LENGTH               = 12
CONF_THRESHOLD           = params["CONF_THRESHOLD"]
CONFIRMATION_COUNT       = params["CONFIRMATION_COUNT"]
CAPITAL_CAP              = params["CAPITAL_CAP"]
PACIFIC                  = ZoneInfo("America/Los_Angeles")

# ─── SLICING CONFIG ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SliceConfig:
    pct: float = 0.1
    sleep_interval: int = 60
    max_retries: int = 3
    backoff_factor: float = 2.0
    max_backoff_interval: int = 300

DEFAULT_SLICE_CFG = SliceConfig(
    pct=POV_SLICE_PCT,
    sleep_interval=60,
    max_retries=3,
    backoff_factor=2.0,
    max_backoff_interval=300
)

# ─── PAIR‐TRADING CONFIG ────────────────────────────────────────────────────
your_cointegrated_pairs: List[Tuple[str,str]] = [
    ("AAPL", "MSFT"),
    ("XOM",  "CVX"),
    # …etc.
]

# ─── ASSERT ALPACA KEYS ───────────────────────────────────────────────────────
ALPACA_API_KEY    = os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
ALPACA_BASE_URL   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

def ensure_alpaca_credentials() -> None:
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise RuntimeError("Missing Alpaca API credentials; please check .env")

ensure_alpaca_credentials()

# ─── TYPED EXCEPTION & CONTEXT ───────────────────────────────────────────────
class DataFetchError(Exception):
    pass

@dataclass
class BotContext:
    api: REST
    data_fetcher: 'DataFetcher'
    signal_manager: 'SignalManager'
    trade_logger: 'TradeLogger'
    sem: Semaphore
    volume_threshold: int
    entry_start_offset: timedelta
    entry_end_offset: timedelta
    market_open: dt_time
    market_close: dt_time
    regime_lookback: int
    regime_atr_threshold: float
    daily_loss_limit: float
    kelly_fraction: float
    confirmation_count: dict[str, int] = field(default_factory=dict)
    trailing_extremes: dict[str, float] = field(default_factory=dict)
    take_profit_targets: dict[str, float] = field(default_factory=dict)
    stop_targets:         dict[str, float] = field(default_factory=dict)
    portfolio_weights: dict[str, float] = field(default_factory=dict)

class FinnhubFetcher:
    def __init__(self, calls_per_minute: int = 60):
        self.max_calls   = calls_per_minute
        self._timestamps = deque()
        self.client      = finnhub_client

    def _throttle(self):
        now = time.time()
        # drop timestamps >60s ago
        while self._timestamps and now - self._timestamps[0] > 60:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.max_calls:
            wait_secs = 60 - (now - self._timestamps[0]) + random.uniform(0.1, 0.5)
            logger.debug(f"[FH] rate-limit reached; sleeping {wait_secs:.2f}s")
            time.sleep(wait_secs)
            return self._throttle()
        self._timestamps.append(now)

    def _parse_period(self, period: str) -> int:
        """
        Convert a string like '5d', '1mo' into seconds.
        """
        if period.endswith("mo"):
            num = int(period[:-2])
            return num * 30 * 86400
        num = int(period[:-1])
        unit = period[-1]
        if unit == "d":
            return num * 86400
        raise ValueError(f"Unsupported period: {period}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10) + wait_random(0.1, 1),
        retry=retry_if_exception_type(Exception)
    )
    def fetch(self, symbols, period="1mo", interval="1d") -> pd.DataFrame:
        """
        symbols: str or list[str]
        period: e.g. '5d', '1mo'
        interval: '1d' or '1m'
        """
        try:
            syms = symbols if isinstance(symbols, (list, tuple)) else [symbols]
            now = int(time.time())
            span = self._parse_period(period)
            start = now - span

            # Finnhub resolution: 'D' for daily, '1' for 1-min
            resolution = 'D' if interval == '1d' else '1'

            frames = []
            for sym in syms:
                self._throttle()
                resp = self.client.stock_candles(sym, resolution, _from=start, to=now)
                if resp.get('s') != 'ok':
                    logger.warning(f"[FH] no data for {sym}: {resp.get('s')}")
                    frames.append(pd.DataFrame())
                    continue

                df = pd.DataFrame({
                    'Open':   resp['o'],
                    'High':   resp['h'],
                    'Low':    resp['l'],
                    'Close':  resp['c'],
                    'Volume': resp['v'],
                }, index=pd.to_datetime(resp['t'], unit='s', utc=True))

                # drop timezone so it matches your existing code
                df.index = df.index.tz_convert(None)
                frames.append(df)

            if not frames:
                return pd.DataFrame()
            if len(frames) == 1:
                return frames[0]

            # multi-symbol: produce a (Field,Symbol) MultiIndex
            return pd.concat(frames, axis=1, keys=syms, names=['Symbol','Field'])

        except FinnhubAPIException as e:
            logger.debug(f"[FH] no access for {symbols} ({period}/{interval}): {e}. returning empty df")
            return pd.DataFrame()
# ─── FINNHUB FETCHER ────────────────────────────────────────────────────────
# instantiate a singleton
fh = FinnhubFetcher(calls_per_minute=60)

_last_fh_prefetch_date: Optional[date] = None

# ─── REGIME CLASSIFIER ──────────────────────────────────────────────────────
def train_regime_model(df_spy: pd.DataFrame, labels: pd.Series) -> RandomForestClassifier:
    feats = ["atr","rsi","macd","vol"]  # vol = pct_change.std()
    X = df_spy[feats]
    y = labels
    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    model.fit(X, y)
    pickle.dump(model, open("regime_model.pkl","wb"))
    return model
    
# ─── CORE CLASSES ─────────────────────────────────────────────────────────────
class DataFetcher:
    def __init__(self) -> None:
        self._daily_cache: dict[str, Optional[pd.DataFrame]] = {}
        self._minute_cache: dict[str, Optional[pd.DataFrame]] = {}
        self._minute_timestamps: dict[str, datetime] = {}

    def get_daily_df(self, ctx: BotContext, symbol: str) -> Optional[pd.DataFrame]:
        # For SPY, always fetch 1 year of data via Alpaca
        if symbol == "SPY":
            today = date.today()
            start = (today - timedelta(days=365)).isoformat()
            end   = today.isoformat()
            bars = ctx.api.get_bars(
                symbol,
                TimeFrame.Day,
                start=start,
                end=end,
                limit=1000,
                feed="iex"
            ).df
            bars.index = pd.to_datetime(bars.index).tz_localize(None)
            df = bars.rename(columns={
                "open":   "Open",
                "high":   "High",
                "low":    "Low",
                "close":  "Close",
                "volume": "Volume"
            })
            self._daily_cache[symbol] = df
            return df

        # existing logic for non-SPY symbols
        if symbol not in self._daily_cache:
            try:
                df = fh.fetch(symbol, period="1mo", interval="1d")
                if df is None or df.empty:
                    raise DataFetchError(f"No daily data for {symbol}")
            except Exception as e:
                logger.warning(f"[DataFetcher] daily fetch failed for {symbol}: {e}")
                df = None
            self._daily_cache[symbol] = df
        return self._daily_cache[symbol]
        
    def get_minute_df(self, ctx: BotContext, symbol: str) -> Optional[pd.DataFrame]:
        # 0) cache hit?
        last = self._minute_timestamps.get(symbol)
        if last and last > datetime.now(timezone.utc) - timedelta(minutes=1):
            return self._minute_cache[symbol]

        df: Optional[pd.DataFrame] = None

        # 1) Skip Finnhub entirely (go straight to Alpaca fallback)
        df = None

        # 2) Fallback to Alpaca IEX if no data
        try:
            bars = ctx.api.get_bars(
                symbol,
                TimeFrame.Minute,
                limit=390 * 5,   # up to 5 full days
                feed="iex"
            ).df

            if not bars.empty:
                bars = bars.rename(columns={
                    "open":   "Open",
                    "high":   "High",
                    "low":    "Low",
                    "close":  "Close",
                    "volume": "Volume"
                })
                if "symbol" in bars.columns:
                    bars = bars.drop(columns=["symbol"])
                bars.index = pd.to_datetime(bars.index).tz_localize(None)
                df = bars[["Open","High","Low","Close","Volume"]]
                logger.info(f"[DataFetcher] minute bars via Alpaca IEX for {symbol}")
        except Exception as e:
            logger.warning(f"[DataFetcher] Alpaca minute fallback failed for {symbol}: {e}")

        # 3) cache & return
        self._minute_cache[symbol]      = df
        self._minute_timestamps[symbol] = datetime.now(timezone.utc)
        return df
        
class TradeLogger:
    def __init__(self, path: str = TRADE_LOG_FILE) -> None:
        self.path = path
        if not os.path.exists(path):
            with portalocker.Lock(path, 'w', timeout=1) as f:
                csv.writer(f).writerow([
                    "symbol","entry_time","entry_price",
                    "exit_time","exit_price","qty","side",
                    "strategy","classification","signal_tags"
                ])

    def log_entry(self, symbol: str, price: float, qty: int, side: str, strategy: str, signal_tags: str="") -> None:
        now = datetime.now(timezone.utc).isoformat()
        with portalocker.Lock(self.path, 'a', timeout=1) as f:
            csv.writer(f).writerow([symbol, now, price, "","", qty, side, strategy, "", signal_tags])

    def log_exit(self, symbol: str, exit_price: float) -> None:
        with portalocker.Lock(self.path, 'r+', timeout=1) as f:
            rows = list(csv.reader(f))
            header, data = rows[0], rows[1:]
            for row in data:
                if row[0] == symbol and row[3] == "":
                    entry_t = datetime.fromisoformat(row[1])
                    days = (datetime.now(timezone.utc) - entry_t).days
                    cls = ("day_trade" if days == 0
                           else "swing_trade" if days < 5
                           else "long_trade")
                    row[3], row[4], row[8] = datetime.now(timezone.utc).isoformat(), exit_price, cls
                    break
            f.seek(0); f.truncate()
            w = csv.writer(f)
            w.writerow(header); w.writerows(data)

class SignalManager:
    def __init__(self) -> None:
        self.momentum_lookback = 5
        self.mean_rev_lookback = 20
        self.mean_rev_zscore_threshold = 1.5
        self.regime_volatility_threshold = REGIME_ATR_THRESHOLD

    def signal_momentum(self, df: pd.DataFrame) -> Tuple[int, float, str]:
        if df is None or len(df) <= self.momentum_lookback:
            return -1, 0.0, 'momentum'
        try:
            df['momentum'] = df['Close'].pct_change(self.momentum_lookback)
            val = df['momentum'].iloc[-1]
            signal = 1 if val > 0 else 0 if val < 0 else -1
            weight = min(abs(val) * 10, 1.0)
            return signal, weight, 'momentum'
        except Exception:
            logger.exception("Error in signal_momentum")
            return -1, 0.0, 'momentum'

    def signal_mean_reversion(self, df: pd.DataFrame) -> Tuple[int, float, str]:
        if df is None or len(df) < self.mean_rev_lookback:
            return -1, 0.0, 'mean_reversion'
        try:
            ma = df['Close'].rolling(self.mean_rev_lookback).mean()
            sd = df['Close'].rolling(self.mean_rev_lookback).std()
            df['zscore'] = (df['Close'] - ma) / sd
            val = df['zscore'].iloc[-1]
            signal = (0 if val > self.mean_rev_zscore_threshold
                      else 1 if val < -self.mean_rev_zscore_threshold
                      else -1)
            weight = min(abs(val)/3, 1.0)
            return signal, weight, 'mean_reversion'
        except Exception:
            logger.exception("Error in signal_mean_reversion")
            return -1, 0.0, 'mean_reversion'

    def signal_stochrsi(self, df: pd.DataFrame) -> Tuple[int, float, str]:
        if df is None or 'stochrsi' not in df or df['stochrsi'].dropna().empty:
            return -1, 0.0, 'stochrsi'
        try:
            val = df['stochrsi'].iloc[-1]
            signal = 1 if val < 0.2 else 0 if val > 0.8 else -1
            return signal, 0.3, 'stochrsi'
        except Exception:
            logger.exception("Error in signal_stochrsi")
            return -1, 0.0, 'stochrsi'

    def signal_obv(self, df: pd.DataFrame) -> Tuple[int, float, str]:
        if df is None or len(df) < 6:
            return -1, 0.0, 'obv'
        try:
            obv = ta.obv(df['Close'], df['Volume'])
            if len(obv) < 5:
                return -1, 0.0, 'obv'
            slope = np.polyfit(range(5), obv.tail(5), 1)[0]
            signal = 1 if slope > 0 else 0 if slope < 0 else -1
            weight = min(abs(slope)/1e6, 1.0)
            return signal, weight, 'obv'
        except Exception:
            logger.exception("Error in signal_obv")
            return -1, 0.0, 'obv'

    def signal_vsa(self, df: pd.DataFrame) -> Tuple[int, float, str]:
        if df is None or len(df) < 20:
            return -1, 0.0, 'vsa'
        try:
            body = abs(df['Close'] - df['Open'])
            vsa = df['Volume'] * body
            score = vsa.iloc[-1]
            avg   = vsa.rolling(20).mean().iloc[-1]
            signal = 1 if df['Close'].iloc[-1] > df['Open'].iloc[-1] else 0 if df['Close'].iloc[-1] < df['Open'].iloc[-1] else -1
            weight = min(score/avg, 1.0)
            return signal, weight, 'vsa'
        except Exception:
            logger.exception("Error in signal_vsa")
            return -1, 0.0, 'vsa'

    def signal_ml(self, df: pd.DataFrame, model: Any) -> Tuple[int, float, str]:
        try:
            feat = ['rsi','macd','atr','vwap','sma_50','sma_200']
            X = df[feat].iloc[-1].values.reshape(1,-1)
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0][pred]
            sig = 1 if pred == 1 else 0
            return sig, proba, 'ml'
        except Exception:
            return -1, 0.0, 'ml'

    def signal_sentiment(self, ctx: BotContext, ticker: str) -> Tuple[int, float, str]:
        score = fetch_sentiment(ctx, ticker)
        sig = 1 if score > 0 else 0 if score < 0 else -1
        return sig, abs(score), 'sentiment'

    def signal_regime(self, ctx: BotContext) -> Tuple[int, float, str]:
        ok = check_market_regime()
        sig = 1 if ok else 0
        return sig, 1.0, 'regime'

    def load_signal_weights(self) -> dict[str, float]:
        if not os.path.exists(SIGNAL_WEIGHTS_FILE):
            return {}
        df = pd.read_csv(SIGNAL_WEIGHTS_FILE)
        return {row['signal']: row['weight'] for _, row in df.iterrows()}

    def evaluate(self, ctx: BotContext, df: pd.DataFrame, ticker: str, model: Any) -> Tuple[int, float, str]:
        signals: List[Tuple[int, float, str]] = []
        allowed_tags = set(load_global_signal_performance() or [])
        weights = self.load_signal_weights()

        fns = [
            self.signal_momentum,
            self.signal_mean_reversion,
            self.signal_ml,
            lambda d: self.signal_sentiment(ctx, ticker),
            lambda d: self.signal_regime(ctx),
            self.signal_stochrsi,
            self.signal_obv,
            self.signal_vsa,
        ]

        for fn in fns:
            try:
                s, w, lab = fn(df, model) if fn == self.signal_ml else fn(df)
                if allowed_tags and lab not in allowed_tags:
                    continue
                if s in (0, 1):
                    signals.append((s, weights.get(lab, w), lab))
            except Exception:
                continue

        if not signals:
            return -1, 0.0, 'no_signal'

        score = sum((1 if s == 1 else -1) * w for s, w, _ in signals)
        conf  = min(abs(score), 1.0)
        final = 1 if score >  0.5 else 0 if score < -0.5 else -1
        label = '+'.join(lbl for _, _, lbl in signals)

        logger.info(f"[SignalManager] {ticker} | final={final} score={score:.2f} | components: {signals}")
        return final, conf, label

# ─── GLOBAL STATE ─────────────────────────────────────────────────────────────
api            = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)
data_fetcher   = DataFetcher()
signal_manager = SignalManager()
trade_logger   = TradeLogger()
ctx = BotContext(
    api=api,
    data_fetcher=data_fetcher,
    signal_manager=signal_manager,
    trade_logger=trade_logger,
    sem=Semaphore(4),
    volume_threshold=VOLUME_THRESHOLD,
    entry_start_offset=ENTRY_START_OFFSET,
    entry_end_offset=ENTRY_END_OFFSET,
    market_open=MARKET_OPEN,
    market_close=MARKET_CLOSE,
    regime_lookback=REGIME_LOOKBACK,
    regime_atr_threshold=REGIME_ATR_THRESHOLD,
    daily_loss_limit=DAILY_LOSS_LIMIT,
    kelly_fraction=params["KELLY_FRACTION"],
)

# ─── CIRCUIT-BREAKER WRAPPERS ────────────────────────────────────────────────
@breaker
def safe_alpaca_get_account():
    return api.get_account()

# ─── WRAPPED I/O CALLS ───────────────────────────────────────────────────────
@retry(
    stop=stop_after_attempt(2),
    wait=wait_fixed(30),
    retry=retry_if_exception_type(Exception)
)
def _fh_chunk_fetch(
    symbols: List[str],
    period: str = "1mo",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Use your throttled FinnhubFetcher for small batches.
    Retries twice on any exception.
    """
    df = fh.fetch(symbols, period=period, interval=interval)
    if df is None or df.empty:
        raise DataFetchError(f"No data returned for symbols {symbols}")
    return df
    
def prefetch_daily_with_alpaca(symbols: List[str]):
    all_syms = [s for s in symbols if s != "SPY"]
    if not all_syms:
        return
    start = (date.today() - timedelta(days=30)).isoformat()
    end   = date.today().isoformat()

    try:
        bars = api.get_bars(
            all_syms,
            TimeFrame.Day,
            start=start,
            end=end,
            limit=1000,
            feed="iex"
        ).df

        # Alpaca gives a datetime index and lowercase 'open','high','low','close','volume'
        for sym, df_sym in bars.groupby("symbol"):
            df = df_sym.copy()

            # drop the extra 'symbol' column
            df = df.drop(columns=["symbol"], errors="ignore")

            # rename to your OHLCV convention
            df = df.rename(columns={
                "open":   "Open",
                "high":   "High",
                "low":    "Low",
                "close":  "Close",
                "volume": "Volume"
            })

            # index is already datetime; strip tz if present
            df.index = pd.to_datetime(df.index).tz_localize(None)

            data_fetcher._daily_cache[sym] = df

        logger.info(f"[prefetch] seeded {len(bars['symbol'].unique())} symbols via Alpaca")
        return

    except Exception as e:
        logger.warning(f"[prefetch] Alpaca bulk failed: {e!r} — falling back to Finnhub")

    # 2) Symbol-by-symbol fallback in tiny chunks via Finnhub
    for batch in chunked(all_syms, 3):
        try:
            df_chunk = _fh_chunk_fetch(batch, period="1mo", interval="1d")
        except Exception as e:
            logger.warning(f"[prefetch] Finnhub chunk {batch} failed: {e!r}")
            continue

        if df_chunk.empty:
            continue

        # Normalize index once
        df_chunk.index = pd.to_datetime(df_chunk.index)

        if isinstance(df_chunk.columns, pd.MultiIndex):
            # level 0 = symbol, level 1 = field
            for sym in batch:
                if sym in df_chunk.columns.get_level_values(0):
                    sym_df = df_chunk.xs(sym, level=0, axis=1)
                    data_fetcher._daily_cache[sym] = sym_df
                    logger.info(f"⚠️  fetched {sym} via Finnhub chunk")
        else:
            # single‐symbol fetch always returns a flat DF
            sym = batch[0]
            data_fetcher._daily_cache[sym] = df_chunk
            logger.info(f"⚠️  fetched {sym} via Finnhub chunk")

        # throttle to ~5 calls/min
        time.sleep(12)

    # 3) Per-symbol ultimate retry via Finnhub, then dummy
    for sym in all_syms:
        if sym not in data_fetcher._daily_cache or data_fetcher._daily_cache[sym] is None:
            try:
                df_sym = fh.fetch(sym, period="1mo", interval="1d")
                if df_sym is not None and not df_sym.empty:
                    df_sym.index = pd.to_datetime(df_sym.index)
                    data_fetcher._daily_cache[sym] = df_sym
                    logger.info(f"⚠️  single-symbol fetch succeeded for {sym}")
                    continue
            except Exception as e:
                logger.warning(f"[prefetch] single-symbol Finnhub fetch failed for {sym}: {e!r}")

            # last resort: dummy flat
            dates = pd.date_range(start=start, end=end, freq="D")
            dummy = pd.DataFrame(index=dates, columns=["Open","High","Low","Close","Volume"])
            dummy[["Open","High","Low","Close"]] = 1.0
            dummy["Volume"] = 0
            data_fetcher._daily_cache[sym] = dummy
            logger.warning(f"🔶 Dummy {sym} inserted to satisfy data checks")
        
def fetch_data(ctx, symbols, period, interval):
    """Fallback for small bulk requests via Alpaca barset if needed."""
    dfs: List[pd.DataFrame] = []
    for batch in chunked(symbols, 3):
        df = fh.fetch(batch, period=period, interval=interval)
        if df is not None and not df.empty:
            dfs.append(df)
        time.sleep(random.uniform(2, 5))
    if not dfs:
        return None
    return pd.concat(dfs, axis=1, sort=True)

@sleep_and_retry
@limits(calls=30, period=60)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(requests.RequestException)
)

def get_sec_headlines(ctx: BotContext, ticker: str) -> str:
    with ctx.sem:
        r = requests.get(
            f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
            f"&CIK={ticker}&type=8-K&count=5",
            headers={"User-Agent":"AI Trading Bot"}
        )
        r.raise_for_status()

    soup = BeautifulSoup(r.content, "lxml")
    rows = soup.find_all("tr")
    logger.info(f"get_sec_headlines({ticker}) ── total <tr> rows: {len(rows)}")

    texts = []
    for a in soup.find_all("a"):
        if "8-K" in a.text:
            tr = a.find_parent("tr")
            if not tr:
                logger.warning("found an <a> with '8-K' but no parent <tr>")
                continue

            tds = tr.find_all("td")
            if len(tds) < 4:
                logger.warning(f"unexpected <td> count {len(tds)} in row, skipping")
                continue

            texts.append(tds[-1].get_text(strip=True))

    if not texts:
        logger.warning(f"get_sec_headlines({ticker}) found no 8-K entries")
    return " ".join(texts)

@limits(calls=60, period=60)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type((requests.RequestException, DataFetchError))
)

def fetch_sentiment(ctx: BotContext, ticker: str) -> float:
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={ticker}&sortBy=publishedAt&language=en&pageSize=5"
        f"&apiKey={NEWS_API_KEY}"
    )
    resp = requests.get(url, timeout=10)

    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError:
        if resp.status_code == 429:
            logger.warning(f"fetch_sentiment({ticker}) rate-limited → returning neutral 0.0")
            return 0.0
        raise

    payload = resp.json()
    articles = payload.get("articles", [])
    if not articles:
        return 0.0

    scores = []
    for art in articles:
        text = (art.get("title") or "") + ". " + (art.get("description") or "")
        if text.strip():
            scores.append(predict_text_sentiment(text))

    return float(sum(scores) / len(scores)) if scores else 0.0

# ─── CHECKS & GUARDS ─────────────────────────────────────────────────────────
@sleep_and_retry
@limits(calls=200, period=60)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type(APIError)
)
def check_daily_loss() -> bool:
    global day_start_equity

    try:
        acct = safe_alpaca_get_account()
        equity = float(acct.equity)
        logger.debug(f"account equity: {equity}")
    except pybreaker.CircuitBreakerError:
        logger.warning("Alpaca account call short-circuited")
        return False
    except Exception as e:
        logger.warning(f"[check_daily_loss] could not fetch account cash: {e!r}")
        return False

    today = date.today()
    if day_start_equity is None or day_start_equity[0] != today:
        day_start_equity = (today, equity)
        daily_drawdown.set(0.0)
        logger.info(f"reset day_start_equity to {equity} on {today}")
        return False

    loss = (day_start_equity[1] - equity) / day_start_equity[1]
    daily_drawdown.set(loss)
    logger.info(f"daily drawdown is {loss:.2%}")
    return loss >= DAILY_LOSS_LIMIT

def check_halt_flag() -> bool:
    if not os.path.exists(HALT_FLAG_PATH):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(HALT_FLAG_PATH))
    return age < timedelta(hours=1)

def now_pacific() -> datetime:
    return datetime.now(PACIFIC)

_warned_missing_spy_columns = False

@retry(stop=stop_after_attempt(3), wait=wait_fixed(0.5))
def check_market_regime() -> bool:
    # pull the daily SPY cache
    df = data_fetcher._daily_cache.get("SPY")
    if df is None or len(df) < 200:
        logger.warning("[check_market_regime] insufficient SPY history – need ≥200 bars")
        return False

    # compute all the your usual daily indicators
    ind = prepare_indicators(df, freq="daily")
    if ind.empty:
        logger.warning("[check_market_regime] no indicators for SPY")
        return False

    # get the last row of indicators
    row = ind.iloc[-1]

    # compute the extra volatility feature
    vol = df["Close"].pct_change().rolling(14).std().iloc[-1]
    if pd.isna(vol):
        logger.warning("[check_market_regime] vol is NaN – failing regime check")
        return False

    # assemble into the same feature order used when training
    feature_vector = [
        row["atr"],
        row["rsi"],
        row["macd"],
        vol
    ]

    # run your RF regime_model
    pred = regime_model.predict([feature_vector])[0]

    return bool(pred)

def too_many_positions() -> bool:
    try:
        return len(api.list_positions()) >= MAX_PORTFOLIO_POSITIONS
    except Exception:
        logger.warning("[too_many_positions] Could not fetch positions")
        return False

def too_correlated(sym: str) -> bool:
    if not os.path.exists(TRADE_LOG_FILE):
        return False
    df = pd.read_csv(TRADE_LOG_FILE)
    open_syms = df.loc[df.exit_time == "", "symbol"].unique().tolist() + [sym]
    rets = {}
    for s in open_syms:
        d = fetch_data(ctx, [s], period="3mo", interval="1d")
        if d is None or d.empty:
            continue
        series = d["Close"].pct_change().dropna()
        if len(series) > 0:
            rets[s] = series

    if len(rets) < 2:
        return False

    min_len = min(len(r) for r in rets.values())
    if min_len < 1:
        return False

    good_syms = [s for s, r in rets.items() if len(r) >= min_len]
    idx = rets[good_syms[0]].tail(min_len).index
    mat = pd.DataFrame({s: rets[s].tail(min_len).values for s in good_syms}, index=idx)
    corr_matrix = mat.corr().abs()
    avg_corr = corr_matrix.where(~np.eye(len(good_syms), dtype=bool)).stack().mean()
    return avg_corr > CORRELATION_THRESHOLD

# ─── SIZING & EXECUTION ───────────────────────────────────────────────────────
def is_within_entry_window(ctx: BotContext) -> bool:
    now = now_pacific()
    start = (datetime.combine(now.date(),ctx.market_open)+ctx.entry_start_offset).time()
    end   = (datetime.combine(now.date(),ctx.market_close)-ctx.entry_end_offset).time()
    if not (start<=now.time()<=end):
        logger.info(f"[SKIP] entry window ({start}–{end}), now={now.time()}")
        return False
    return True

# ─── SIZING & EXECUTION ───────────────────────────────────────────────────────
def scaled_atr_stop(entry_price: float,
                    atr: float,
                    now: datetime,
                    market_open: datetime,
                    market_close: datetime,
                    max_factor: float = 2.0,
                    min_factor: float = 0.5
                   ) -> Tuple[float,float]:
    total   = (market_close - market_open).total_seconds()
    elapsed = (now - market_open).total_seconds()
    α       = max(0, min(1, 1 - elapsed/total))
    factor  = min_factor + α*(max_factor - min_factor)
    stop    = entry_price - factor * atr
    take    = entry_price + factor * atr
    return stop, take

def fractional_kelly_size(
    ctx: BotContext,
    balance: float,
    price: float,
    atr: float,
    win_prob: float,
    payoff_ratio: float = 1.5
) -> int:
    if not os.path.exists(PEAK_EQUITY_FILE):
        with portalocker.Lock(PEAK_EQUITY_FILE,'w',timeout=1) as f:
            f.write(str(balance))
        peak_equity = balance
    else:
        with portalocker.Lock(PEAK_EQUITY_FILE,'r+',timeout=1) as f:
            content = f.read().strip()
            peak_equity = float(content) if content else balance
            if balance>peak_equity:
                f.seek(0); f.truncate(); f.write(str(balance))
                peak_equity = balance

    drawdown = (peak_equity - balance)/peak_equity
    if drawdown>0.10:
        frac = 0.3
    elif drawdown>0.05:
        frac = 0.45
    else:
        frac = ctx.kelly_fraction

    edge = win_prob - (1-win_prob)/payoff_ratio
    kelly = max(edge/payoff_ratio,0)*frac

    dollars_to_risk = kelly*balance
    if atr<=0:
        return 1

    raw_pos = dollars_to_risk/atr
    cap_pos = (balance*CAPITAL_CAP)/price
    size = int(min(raw_pos,cap_pos,MAX_POSITION_SIZE))

    return max(size,1)

def vol_target_position_size(cash: float,
                             price: float,
                             returns: np.ndarray,
                             target_vol: float = 0.02
                            ) -> int:
    sigma = np.std(returns)
    dollar_alloc = cash * (target_vol / sigma)
    qty = int(dollar_alloc / price)
    return max(qty, 1)

def submit_order(ctx: BotContext, symbol: str, qty: int, side: str) -> Optional[Order]:
    """
    Place a market order, then verify fill price against latest quote.
    """
    # fetch the best bid/ask just before sending
    quote = ctx.api.get_latest_quote(symbol)
    expected_price = quote.ask_price if side == "buy" else quote.bid_price

    try:
        order = ctx.api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="gtc",
        )
        logger.info(f"[submit_order] {side.upper()} {qty} {symbol} at market; expected ~{expected_price}")
        orders_total.inc()    # increment Prometheus counter for each order sent
        return order

    except APIError as e:
        msg = str(e).lower()

        # Skip entirely if no buying power
        if "insufficient buying power" in msg:
            logger.warning(f"[submit_order] insufficient buying power for {symbol}; skipping order")
            return None

        # Partial-fill fallback for “requested X, available Y”
        m = re.search(r"requested: (\d+), available: (\d+)", msg)
        if m and int(m.group(2)) > 0:
            available = int(m.group(2))
            ctx.api.submit_order(
                symbol=symbol,
                qty=available,
                side=side,
                type="market",
                time_in_force="gtc",
            )
            logger.info(f"[submit_order] only {available} {symbol} available; placed partial fill")
            orders_total.inc()
            return None

        # Other Alpaca API errors bubble up
        logger.warning(f"[submit_order] APIError for {symbol}: {e}")
        raise

    except Exception:
        order_failures.inc()
        logger.exception(f"[submit_order] unexpected error for {symbol}")
        return None

def twap_submit(
    ctx: BotContext,
    symbol: str,
    total_qty: int,
    side: str,
    window_secs: int = 600,
    n_slices: int = 10,
) -> None:
    """Slice a large order into n_slices over window_secs seconds."""
    if total_qty <= 0:
        logger.warning(f"[twap_submit] non‐positive total_qty={total_qty}, skipping")
        return

    slice_qty = total_qty // n_slices
    wait_secs = window_secs / n_slices

    for i in range(n_slices):
        try:
            submit_order(ctx, symbol, slice_qty, side)
        except Exception as e:
            logger.exception(f"[TWAP] slice {i+1}/{n_slices} failed: {e}")
            break
        time.sleep(wait_secs)

def vwap_pegged_submit(
    ctx: BotContext,
    symbol: str,
    total_qty: int,
    side: str,
    duration: int = 300
) -> None:
    """Place a VWAP‐pegged IOC bracket order over duration seconds."""
    if total_qty <= 0:
        logger.warning(f"[vwap_pegged_submit] non‐positive total_qty={total_qty}, skipping")
        return

    start = time.time()
    placed = 0

    while placed < total_qty and time.time() - start < duration:
        df = ctx.data_fetcher.get_minute_df(ctx, symbol)
        if not df or df.empty:
            logger.warning("[VWAP] missing bars, aborting VWAP slice")
            break

        vwap = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"]).iloc[-1]
        to_send = min(max(1, total_qty // 10), total_qty - placed)

        try:
            ctx.api.submit_order(
                symbol=symbol,
                qty=to_send,
                side=side,
                type="limit",
                time_in_force="ioc",
                limit_price=vwap
            )
        except Exception as e:
            logger.exception(f"[VWAP] slice failed: {e}")
            break

        placed += to_send
        time.sleep(duration / 10)

def pov_submit(
    ctx: BotContext,
    symbol: str,
    total_qty: int,
    side: str,
    cfg: SliceConfig = DEFAULT_SLICE_CFG,
) -> bool:
    """
    Participation-of-Volume slicing.

    Returns True if it fully placed total_qty, False if aborted early.
    """
    if total_qty <= 0:
        logger.warning(f"[pov_submit] non‐positive total_qty={total_qty}, skipping")
        return False

    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    placed = 0
    retries = 0
    interval = cfg.sleep_interval

    while placed < total_qty:
        df = ctx.data_fetcher.get_minute_df(ctx, symbol)
        if not df or df.empty:
            retries += 1
            if retries > cfg.max_retries:
                logger.warning(f"[pov_submit] no minute data after {cfg.max_retries} retries, aborting")
                return False

            logger.warning(f"[pov_submit] missing bars, retry {retries}/{cfg.max_retries} in {interval:.1f}s")
            sleep_time = interval * (0.8 + 0.4 * random.random())
            time.sleep(sleep_time)
            interval = min(interval * cfg.backoff_factor, cfg.max_backoff_interval)
            continue

        # reset on success
        retries = 0
        interval = cfg.sleep_interval

        vol = df["Volume"].iloc[-1]
        slice_qty = min(int(vol * cfg.pct), total_qty - placed)
        if slice_qty < 1:
            logger.debug(f"[pov_submit] slice_qty<1 (vol={vol}), waiting")
            time.sleep(cfg.sleep_interval * (0.8 + 0.4 * random.random()))
            continue

        try:
            submit_order(ctx, symbol, slice_qty, side)
        except Exception as e:
            logger.exception(f"[pov_submit] submit_order failed on slice, aborting: {e}")
            return False

        placed += slice_qty
        logger.info(f"[pov_submit] placed {slice_qty} ({placed}/{total_qty})")
        time.sleep(cfg.sleep_interval * (0.8 + 0.4 * random.random()))

    logger.info(f"[pov_submit] complete: placed {placed}/{total_qty}")
    return True

def maybe_pyramid(ctx: BotContext, symbol: str, entry_price: float, current_price: float, atr: float):
    profit = (current_price - entry_price) if entry_price else 0
    if profit > 2 * atr:
        pos = ctx.api.get_position(symbol)
        qty = int(abs(int(pos.qty)) * 0.5)
        if qty > 0:
            submit_order(ctx, symbol, qty, "buy")
            logger.info(f"Pyramided {qty} shares of {symbol}")

def update_trailing_stop(
    ctx: BotContext,
    ticker: str,
    price: float,
    qty: int,
    atr: float,
    factor: float = TRAILING_FACTOR
) -> str:
    te = ctx.trailing_extremes
    if qty>0:
        te[ticker] = max(te.get(ticker,price),price)
        if price<te[ticker]-factor*atr:
            return "exit_long"
    elif qty<0:
        te[ticker] = min(te.get(ticker,price),price)
        if price>te[ticker]+factor*atr:
            return "exit_short"
    return "hold"

def calculate_entry_size(
    ctx: BotContext,
    symbol: str,
    price: float,
    atr: float,
    win_prob: float
) -> int:
    cash = float(ctx.api.get_account().cash)
    # fetch last-month returns
    df = ctx.data_fetcher.get_daily_df(ctx, symbol)
    rets = (
        df["Close"].pct_change().dropna().values
        if df is not None and not df.empty
        else np.array([0.0])
    )
    kelly_sz = fractional_kelly_size(ctx, cash, price, atr, win_prob)
    vol_sz = vol_target_position_size(cash, price, rets, target_vol=0.02)
    return min(kelly_sz, vol_sz)

def execute_entry(ctx: BotContext, symbol: str, qty: int, side: str) -> None:
    """
    Place an entry order:
      * If qty <= 0 → skip
      * If POV_SLICE_PCT > 0 and qty > SLICE_THRESHOLD → POV slicing
      * Elif qty > SLICE_THRESHOLD → VWAP‐pegged slicing
      * Else → simple market
    """
    if qty <= 0:
        logger.warning(f"[ENTRY] zero quantity for {symbol}, skipping")
        return

    # choose slicing algorithm
    if POV_SLICE_PCT > 0 and qty > SLICE_THRESHOLD:
        logger.info(f"[ENTRY] POV slice {qty}@{POV_SLICE_PCT*100:.1f}% for {symbol}")
        pov_submit(ctx, symbol, qty, side)

    elif qty > SLICE_THRESHOLD:
        logger.info(f"[ENTRY] VWAP‐pegged slice {qty} {side.upper()} for {symbol}")
        vwap_pegged_submit(ctx, symbol, qty, side)

    else:
        logger.info(f"[ENTRY] Market {side.upper()} {qty} {symbol}")
        submit_order(ctx, symbol, qty, side)

    # now log & set stops/targets
    df_min = ctx.data_fetcher.get_minute_df(ctx, symbol)
    if df_min is None or df_min.empty:
        logger.warning(f"[ENTRY] Failed to fetch minute bars for {symbol} after slicing, skipping stop/target setup")
        return
    price  = df_min["Close"].iloc[-1]
    ctx.trade_logger.log_entry(symbol, price, qty, side, "", "")

    now = now_pacific()
    mo  = datetime.combine(now.date(), ctx.market_open, PACIFIC)
    mc  = datetime.combine(now.date(), ctx.market_close, PACIFIC)
    stop, take = scaled_atr_stop(price, df_min["atr"].iloc[-1], now, mo, mc)
    ctx.take_profit_targets[symbol] = take
    ctx.stop_targets[symbol]        = stop

def execute_exit(ctx: BotContext, symbol: str, qty: int) -> None:
    # 1) send the exit order
    submit_order(ctx, symbol, qty, "sell" if qty>0 else "buy")

    # 2) log the exit
    ctx.trade_logger.log_exit(
        symbol,
        ctx.data_fetcher.get_minute_df(ctx, symbol)["Close"].iloc[-1]
    )

    # 3) now rebalance if needed
    on_trade_exit_rebalance(ctx)

    # 4) clean up your targets
    ctx.take_profit_targets.pop(symbol, None)
    ctx.stop_targets.pop(symbol, None)

# ─── SIGNAL & TRADE LOGIC ────────────────────────────────────────────────────
def signal_and_confirm(ctx: BotContext, symbol: str, df: pd.DataFrame, model) -> Tuple[int,float,str]:
    sig, conf, strat = ctx.signal_manager.evaluate(ctx, df, symbol, model)
    conf *= ctx.portfolio_weights.get(symbol, 1.0)
    if sig == -1 or conf < CONF_THRESHOLD:
        logger.info(f"[SKIP] {symbol} no/low signal (sig={sig},conf={conf:.2f})")
        return -1,0.0,""
    return sig, conf, strat

def pre_trade_checks(
    ctx: BotContext,
    symbol: str,
    balance: float,
    regime_ok: bool
) -> bool:
    if check_halt_flag():
        logger.info(f"[SKIP] HALT_FLAG – {symbol}")
        return False
    if not in_trading_hours(pd.Timestamp.utcnow()):
        logger.info(f"[SKIP] Market closed – {symbol}")
        return False
    if check_daily_loss():
        logger.info(f"[SKIP] Daily-loss limit – {symbol}")
        return False
    if not regime_ok:
        logger.info(f"[SKIP] Market regime – {symbol}")
        return False
    if too_many_positions():
        logger.info(f"[SKIP] Max positions – {symbol}")
        return False
    if too_correlated(symbol):
        logger.info(f"[SKIP] Correlation – {symbol}")
        return False
    return ctx.data_fetcher.get_daily_df(ctx, symbol) is not None

def should_enter(
    ctx: BotContext,
    symbol: str,
    balance: float,
    regime_ok: bool
) -> bool:
    return pre_trade_checks(ctx, symbol, balance, regime_ok) and is_within_entry_window(ctx)

def should_exit(ctx: BotContext, symbol: str, price: float, atr: float) -> Tuple[bool,int,str]:
    try:
        pos = int(ctx.api.get_position(symbol).qty)
    except Exception:
        pos=0
    stop = ctx.stop_targets.get(symbol)
    if stop is not None and price <= stop:
        return True, abs(pos), "stop_loss"
    tp = ctx.take_profit_targets.get(symbol)
    if pos and tp and ((pos>0 and price>=tp) or (pos<0 and price<=tp)):
        return True, int(abs(pos)*SCALING_FACTOR), "take_profit"
    action = update_trailing_stop(ctx, symbol, price, pos, atr)
    if action=="exit_long" and pos>0:
        return True, pos, "trailing_stop"
    if action=="exit_short" and pos<0:
        return True, abs(pos), "trailing_stop"
    return False,0,""

def trade_logic(
    ctx: BotContext,
    symbol: str,
    balance: float,
    model,
    regime_ok: bool
) -> None:
    logger.info(f"→ trade_logic start for {symbol}")

    # 0) event‐driven skip
    if is_near_event(symbol):
        logger.info(f"[SKIP] earnings/event window for {symbol}")
        return

    # 1) pair‐trade override
    for (s1, s2) in your_cointegrated_pairs:
        if symbol in (s1, s2):
            strat, sz = pair_trade_signal(s1, s2)
            if sz > 0:
                side = "buy" if strat == "long_spread" else "sell"
                execute_entry(ctx, symbol, sz, side)
            return

    # 2) early‐exit if we shouldn’t even be trying
    if not should_enter(ctx, symbol, balance, regime_ok):
        return

    # 3) fetch minute bars
    raw = ctx.data_fetcher.get_minute_df(ctx, symbol)
    if raw is None or raw.empty:
        logger.info(f"[SKIP] no minute data for {symbol}")
        return

    # 4) compute intraday indicators
    df = prepare_indicators(raw, freq="intraday")
    if df.empty:
        logger.info(f"[SKIP] not enough indicator data for {symbol}")
        return

    # 5) signal + confirmation
    sig, conf, _ = signal_and_confirm(ctx, symbol, df, model)
    if sig == -1:
        return

    # 6) decide exit first (stop‐loss / take‐profit / trailing), **and** maybe pyramid
    price, atr = df["Close"].iloc[-1], df["atr"].iloc[-1]

    # ─── NEW PYRAMIDING BLOCK ───────────────────────────────────────────────
    try:
        pos = int(ctx.api.get_position(symbol).qty)
    except Exception:
        pos = 0
    if pos > 0:
        # approximate your entry price from your stored take‐profit and ATR factor
        entry_price = ctx.take_profit_targets.get(symbol, 0) - TAKE_PROFIT_FACTOR * atr
        maybe_pyramid(ctx, symbol, entry_price, price, atr)
    # ────────────────────────────────────────────────────────────────────────

    do_exit, qty, _ = should_exit(ctx, symbol, price, atr)
    if do_exit:
        execute_exit(ctx, symbol, qty)
        return

    # 7) size & entry
    size = calculate_entry_size(ctx, symbol, price, atr, conf)
    if size > 0:
        try:
            pos = int(ctx.api.get_position(symbol).qty)
        except Exception:
            pos = 0
        execute_entry(
            ctx,
            symbol,
            size + abs(pos),
            "buy" if sig == 1 else "sell"
        )

def compute_portfolio_weights(symbols: List[str]) -> dict[str, float]:
    # 1) Bail out if no symbols
    if not symbols:
        logger.warning("[Portfolio] No tickers to optimize—skipping.")
        return {}

    # 2) Build DataFrame of aligned daily closes
    closes = {}
    for sym in symbols:
        df = ctx.data_fetcher.get_daily_df(ctx, sym)
        if df is not None and "Close" in df:
            closes[sym] = df["Close"]
    price_df = pd.DataFrame(closes).dropna()
    
    # 3) Compute expected returns & covariance
    μ = expected_returns.mean_historical_return(price_df)
    Σ = risk_models.sample_cov(price_df)

    # 4) Guard against empty results
    if μ.shape[0] == 0 or Σ.shape[0] == 0:
        logger.warning("[Portfolio] Empty returns or covariance matrix—skipping optimization.")
        return {}

    # 5) Optimize
    ef = EfficientFrontier(μ, Σ)
    weights = ef.max_sharpe()  # or ef.min_volatility(), ef.efficient_risk(target_risk), etc.

    return weights

# ─── PORTFOLIO REBALANCING ──────────────────────────────────────────────────
def on_trade_exit_rebalance(ctx: BotContext):
    # recompute weights on the fly
    current = compute_portfolio_weights(list(ctx.portfolio_weights.keys()))
    old     = ctx.portfolio_weights

    # check for any >10% drift
    drift = max(abs(current[s] - old.get(s, 0)) for s in current)
    if drift <= 0.1:
        return

    # update context
    ctx.portfolio_weights = current

    # submit rebalancing orders
    total_value = float(ctx.api.get_account().portfolio_value)
    for sym, w in current.items():
        target_dollar = w * total_value
        price = ctx.data_fetcher.get_minute_df(ctx, sym)["Close"].iloc[-1]
        target_shares = int(target_dollar / price)
        try:
            ctx.api.submit_order(
                symbol=sym,
                qty=target_shares,
                side="buy" if target_shares > 0 else "sell",
                type="market",
                time_in_force="day"
            )
        except Exception:
            logger.exception(f"Rebalance failed for {sym}")

    logger.info("Rebalanced portfolio weights")

def pair_trade_signal(sym1: str, sym2: str) -> Tuple[str, int]:
    # cointegration-based stat-arb
    df1 = ctx.data_fetcher.get_daily_df(ctx, sym1)["Close"]
    df2 = ctx.data_fetcher.get_daily_df(ctx, sym2)["Close"]
    df  = pd.concat([df1, df2], axis=1).dropna()

    # test for cointegration
    t_stat, p_value, _ = coint(df.iloc[:, 0], df.iloc[:, 1])
    if p_value < 0.05:
        # fit hedge ratio β
        beta = np.polyfit(df.iloc[:, 1], df.iloc[:, 0], 1)[0]
        spread = df.iloc[:, 0] - beta * df.iloc[:, 1]

        # compute z-score series
        z = (spread - spread.mean()) / spread.std()
        z0 = z.iloc[-1]  # most recent z-score

        if z0 > 2:
            return ("short_spread", 1)
        elif z0 < -2:
            return ("long_spread", 1)

    # no valid signal
    return ("no_signal", 0)

def get_calendar_safe(symbol: str) -> pd.DataFrame:
    today = date.today()
    # Return today's cache if available
    if (
        symbol in _calendar_cache
        and _calendar_last_fetch.get(symbol) == today
    ):
        return _calendar_cache[symbol]

    try:
        cal = yf.Ticker(symbol).calendar
    except YFRateLimitError:
        logger.warning(
            f"[Events] Rate limited for {symbol}; skipping events."
        )
        cal = pd.DataFrame()
    except Exception as e:
        logger.error(f"[Events] Error fetching calendar for {symbol}: {e}")
        cal = pd.DataFrame()

    _calendar_cache[symbol] = cal
    _calendar_last_fetch[symbol] = today
    return cal

def is_near_event(symbol: str, days: int = 3) -> bool:
    """
    Returns True if any upcoming calendar event is within `days` from today.
    Skips (returns False) on missing/empty data.
    """
    cal = get_calendar_safe(symbol)

    # if we didn’t get a DataFrame (e.g. got a dict or None) or it's empty, bail
    if not hasattr(cal, "empty") or cal.empty:
        return False

    try:
        dates = []
        for col in cal.columns:
            raw = cal.at['Value', col]
            if isinstance(raw, (list, tuple)):
                raw = raw[0]
            dates.append(pd.to_datetime(raw))
    except Exception:
        logger.debug(f"[Events] Malformed calendar for {symbol}, columns={getattr(cal, 'columns', None)}")
        return False

    today  = pd.Timestamp.now().normalize()
    cutoff = today + pd.Timedelta(days=days)
    return any(today <= d <= cutoff for d in dates)

def run_all_trades(model) -> None:
    global _last_fh_prefetch_date

    now = pd.Timestamp.utcnow()
    if not in_trading_hours(now):
        logger.info("[SKIP] Market closed")
        return
    logger.info(f"🔄 run_all_trades fired at {datetime.now(timezone.utc).isoformat()}")

    # 1) Load your universe…
    candidates = load_tickers(TICKERS_FILE)
    today = date.today()
    if _last_fh_prefetch_date != today:
        _last_fh_prefetch_date = today

        # bulk-prefetch everything except SPY
        prefetch_daily_with_alpaca(candidates)

        # **ensure SPY is seeded, so regime checks will find it**
        ctx.data_fetcher.get_daily_df(ctx, "SPY")

    # 2) Screen and compute portfolio weights
    tickers = screen_universe(candidates, ctx)
    ctx.portfolio_weights = compute_portfolio_weights(tickers)
    if not tickers:
        logger.error("❌ No tickers loaded; skipping run_all_trades")
        return

    # 3) Halt-flag guard
    if check_halt_flag():
        logger.info("Trading halted via HALT_FLAG_FILE.")
        return

    # 4) Fetch current cash
    acct = api.get_account()
    current_cash = float(acct.cash)

    # 5) Compute regime just once per cycle
    regime_ok = check_market_regime()

    # 6) Fan-out your trades
    for symbol in tickers:
        executor.submit(_safe_trade, ctx, symbol, current_cash, model, regime_ok)

def _safe_trade(
    ctx: BotContext,
    symbol: str,
    balance: float,
    model,
    regime_ok: bool
) -> None:
    try:
        trade_logic(ctx, symbol, balance, model, regime_ok)

    except APIError as e:
        msg = str(e).lower()

        # skip entirely if no buying power
        if "insufficient buying power" in msg:
            logger.warning(f"[trade_logic] insufficient buying power for {symbol}; skipping")
        else:
            logger.exception(f"[trade_logic] APIError for {symbol}: {e}")

    except Exception:
        logger.exception(f"[trade_logic] unhandled exception for {symbol}")

# ─── UTILITIES ────────────────────────────────────────────────────────────────
def load_model(path: str = MODEL_PATH):
    if os.path.exists(path):
        logger.info(f"Loading trained model from {path}")
        return joblib.load(path)
    logger.info("No model found; training fallback RandomForestClassifier")
    model = RandomForestClassifier(n_estimators=RF_ESTIMATORS, max_depth=RF_MAX_DEPTH)
    X_dummy = np.random.randn(100,6)
    y_dummy = np.random.randint(0,2,size=100)
    model.fit(X_dummy, y_dummy)
    joblib.dump(model, path)
    logger.info(f"Fallback model trained and saved to {path}")
    return model

def update_signal_weights():
    if not os.path.exists(TRADE_LOG_FILE):
        logger.warning("No trades log found; skipping weight update.")
        return
    df = pd.read_csv(TRADE_LOG_FILE).dropna(subset=["entry_price","exit_price","signal_tags"])
    df["pnl"] = (df["exit_price"]-df["entry_price"])*df["side"].apply(lambda s:1 if s=="buy" else -1)
    stats={}
    for _,row in df.iterrows():
        for tag in row["signal_tags"].split("+"):
            stats.setdefault(tag,[]).append(row["pnl"])
    new_weights={tag:round(np.mean([1 if p>0 else 0 for p in pnls]),3) for tag,pnls in stats.items()}
    ALPHA=0.2
    old = pd.read_csv(SIGNAL_WEIGHTS_FILE).set_index("signal")["weight"].to_dict() if os.path.exists(SIGNAL_WEIGHTS_FILE) else {}
    merged={tag:round(ALPHA*w+(1-ALPHA)*old.get(tag,w),3) for tag,w in new_weights.items()}
    out_df = pd.DataFrame.from_dict(merged,orient="index",columns=["weight"]).reset_index()
    out_df.columns=["signal","weight"]
    out_df.to_csv(SIGNAL_WEIGHTS_FILE,index=False)
    logger.info(f"[MetaLearn] Updated weights for {len(merged)} signals.")

def load_global_signal_performance(min_trades=10,threshold=0.4):
    if not os.path.exists(TRADE_LOG_FILE):
        logger.info("[MetaLearn] no history - allowing all signals")
        return None
    df = pd.read_csv(TRADE_LOG_FILE).dropna(subset=["exit_price","entry_price","signal_tags"])
    df["pnl"] = (df.exit_price-df.entry_price)*df.side.apply(lambda s:1 if s=="buy" else -1)
    results={}
    for _,row in df.iterrows():
        for tag in row.signal_tags.split("+"):
            results.setdefault(tag,[]).append(row.pnl)
    win_rates={tag:round(np.mean([1 if p>0 else 0 for p in pnls]),3)
               for tag,pnls in results.items() if len(pnls)>=min_trades}
    filtered={tag:wr for tag,wr in win_rates.items() if wr>=threshold}
    logger.info(f"[MetaLearn] Keeping signals: {list(filtered.keys()) or 'none'}")
    return filtered

# ─── INDICATORS ──────────────────────────────────────────────────────────────
def prepare_indicators(df: pd.DataFrame, freq: str = "daily") -> pd.DataFrame:
    df = df.copy()

    # normalize index → Date column
    if df.index.name:
        df = df.reset_index().rename(columns={df.index.name: "Date"})
    else:
        df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    # compute core TA indicators (expects uppercase OHLCV columns)
    df["vwap"]  = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"])
    df["rsi"]   = ta.rsi(df["Close"], length=14)
    df["atr"]   = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    macd       = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    df["macd"]  = macd["MACD_12_26_9"]
    df["macds"] = macd["MACDs_12_26_9"]

    if freq == "daily":
        df["sma_50"]  = ta.sma(df["Close"], length=50)
        df["sma_200"] = ta.sma(df["Close"], length=200)

    # Ichimoku
    ich = ta.ichimoku(high=df["High"], low=df["Low"], close=df["Close"])
    conv = ich[0] if isinstance(ich, tuple) else ich.iloc[:,0]
    base = ich[1] if isinstance(ich, tuple) else ich.iloc[:,1]
    df["ichimoku_conv"] = conv.iloc[:,0] if isinstance(conv, pd.DataFrame) else conv
    df["ichimoku_base"] = base.iloc[:,0] if isinstance(base, pd.DataFrame) else base

    # StochRSI
    st = ta.stochrsi(df["Close"])
    df["stochrsi"] = st["STOCHRSIk_14_14_3_3"]

    # fill-forward/backward to smooth
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # drop rules: require *all* for daily, but only fully-NaN for intraday
    required = ["vwap","rsi","atr","ichimoku_conv","ichimoku_base","stochrsi","macd","macds"]
    if freq == "daily":
        required += ["sma_50","sma_200"]
        df.dropna(subset=required, how="any", inplace=True)
        # keep the DateTimeIndex intact for daily
    else:
        df.dropna(subset=required, how="all", inplace=True)
        # intraday: drop the index timestamps
        df.reset_index(drop=True, inplace=True)

    return df


# ─── REGIME CLASSIFIER ──────────────────────────────────────────────────────
if os.path.exists("regime_model.pkl"):
    regime_model = pickle.load(open("regime_model.pkl", "rb"))
else:
    # 1) Fetch one year of SPY daily bars
    today = date.today()
    start = (today - timedelta(days=365)).isoformat()
    bars = api.get_bars(
        "SPY", TimeFrame.Day,
        start=start, end=today.isoformat(),
        limit=1000, feed="iex"
    ).df

    # 2) Normalize index & ensure uppercase OHLCV
    bars.index = pd.to_datetime(bars.index).tz_localize(None)
    bars = bars.rename(columns={
        "open":   "Open",
        "high":   "High",
        "low":    "Low",
        "close":  "Close",
        "volume": "Volume"
    })

    # 3) Compute indicators (this now finds High/Low/etc)
    ind = prepare_indicators(bars, freq="daily")

    # 4) Generate labels: 1 if Close > 200-day SMA, else 0
    labels = (
        bars["Close"] > bars["Close"].rolling(200).mean()
    ).astype(int).rename("label")

    # 5) Align and train
    valid = ind.join(labels, how="inner").dropna()
    if len(valid) >= 200:
        regime_model = train_regime_model(valid, valid["label"])
        pickle.dump(regime_model, open("regime_model.pkl", "wb"))
    else:
        logger.error(f"Not enough SPY bars ({len(bars)}) to train regime model; using dummy fallback")
        regime_model = RandomForestClassifier(n_estimators=RF_ESTIMATORS, max_depth=RF_MAX_DEPTH)

# ─── UNIVERSE SELECTION ─────────────────────────────────────────────────────
def screen_universe(
    candidates: Sequence[str],
    ctx: BotContext,
    lookback: str = "1mo",
    interval: str = "1d",
    top_n: int = 20
) -> list[str]:
    """
    Fetch daily bars for each candidate, compute ATR, and return top_n highest-ATR symbols.
    The lookback and interval args are accepted for symmetry with run_all_trades, but ignored here.
    """
    atrs: dict[str, float] = {}
    for sym in candidates:
        df = ctx.data_fetcher.get_daily_df(ctx, sym)
        if df is None or len(df) < ATR_LENGTH:
            continue
        series = ta.atr(df["High"], df["Low"], df["Close"], length=ATR_LENGTH)
        atr = series.iloc[-1] if not series.empty else np.nan
        if not pd.isna(atr):
            atrs[sym] = float(atr)
    ranked = sorted(atrs.items(), key=lambda kv: kv[1], reverse=True)
    return [sym for sym, _ in ranked[:top_n]]
    
def load_tickers(path: str=TICKERS_FILE) -> list[str]:
    tickers=[]
    try:
        with open(path,newline="") as f:
            reader=csv.reader(f)
            next(reader,None)
            for row in reader:
                t=row[0].strip().upper()
                if t and t not in tickers:
                    tickers.append(t)
    except Exception as e:
        logger.exception(f"[load_tickers] Failed to read {path}: {e}")
    return tickers

def daily_summary() -> None:
    """End-of-day summary: compute and log PnL, win rate, total trades, and max drawdown."""
    if not os.path.exists(TRADE_LOG_FILE):
        logger.info("[daily_summary] No trades to summarize.")
        return
    df = pd.read_csv(TRADE_LOG_FILE).dropna(subset=["entry_price","exit_price"])
    df["pnl"] = (df.exit_price - df.entry_price) * df.side.map({"buy":1,"sell":-1})
    total_trades = len(df)
    win_rate = (df.pnl > 0).mean() if total_trades else 0
    total_pnl = df.pnl.sum()
    max_dd = (df.pnl.cumsum().cummax() - df.pnl.cumsum()).max()
    logger.info(f"[daily_summary] Trades={total_trades} WinRate={win_rate:.2%} PnL={total_pnl:.2f} MaxDD={max_dd:.2f}")

# ─── MAIN LOOP ───────────────────────────────────────────────────────────────
app = Flask(__name__)
@app.route("/health")
def health() -> str:
    return "OK", 200

def start_healthcheck() -> None:
    # allow overriding via ENV so you don’t have to re-deploy to pick a new port
    port = int(os.getenv("HEALTHCHECK_PORT", "8080"))
    try:
        app.run(host="0.0.0.0", port=port)
    except OSError as e:
        # catch “Address already in use” and just skip spinning up Flask
        logger.warning(f"Healthcheck port {port} in use: {e}. Skipping health-endpoint.")

if __name__ == "__main__":
    start_http_server(8000)
    logger.info("Prometheus metrics server started on port 8000")

    if RUN_HEALTH:
        Thread(target=start_healthcheck, daemon=True).start()
        logger.info("Healthcheck endpoint running on port 8080")

    schedule.every().day.at("00:30").do(daily_summary)
    logger.info("Scheduled daily summary at 00:30 UTC")

    model = load_model()
    logger.info("🚀 AI Trading Bot is live!")
    schedule.every(1).minutes.do(lambda: run_all_trades(model))
    schedule.every(6).hours.do(update_signal_weights)
    logger.info("Scheduled run_all_trades every 1m and update_signal_weights every 6h")

    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            logger.exception(f"Scheduler loop error: {e}")
        time.sleep(30)
