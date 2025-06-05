import logging
import logging.handlers
import os
import csv
import json
import re
import time
import time as pytime
import pathlib
import random
import signal
import sys
import atexit
from datetime import datetime, date, time as dt_time, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Optional, Tuple, Dict, List, Any, Sequence
from threading import Semaphore, Lock, Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from collections import deque

import numpy as np
# Ensure numpy.NaN exists for pandas_ta compatibility
np.NaN = np.nan

import pandas as pd
import pandas_ta as ta
import pandas_market_calendars as mcal

import requests
from bs4 import BeautifulSoup
from flask import Flask
import schedule
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
import portalocker

from alpaca_trade_api.rest import REST, APIError, TimeFrame
from alpaca_trade_api.entity import Order

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.decomposition import PCA
import pickle
import joblib

from dotenv import load_dotenv
import sentry_sdk

from prometheus_client import start_http_server, Counter, Gauge, Histogram
import finnhub
from finnhub import FinnhubAPIException
import pybreaker

def cancel_all_open_orders(ctx: "BotContext") -> None:
    """
    On startup or each run, cancel every Alpaca order whose status is 'open'.
    """
    try:
        open_orders = ctx.api.list_orders(status="open", limit=500)
        for od in open_orders:
            if getattr(od, "status", "").lower() == "open":
                try:
                    ctx.api.cancel_order(od.id)
                except Exception:
                    pass
    except Exception:
        pass

def reconcile_positions(ctx: 'BotContext') -> None:
    """On startup, fetch all live positions and clear any in-memory stop/take targets for assets no longer held."""
    try:
        live_positions = {pos.symbol: int(pos.qty) for pos in ctx.api.list_positions()}
        with targets_lock:
            symbols_with_targets = list(ctx.stop_targets.keys()) + list(ctx.take_profit_targets.keys())
            for symbol in symbols_with_targets:
                if symbol not in live_positions or live_positions[symbol] == 0:
                    ctx.stop_targets.pop(symbol, None)
                    ctx.take_profit_targets.pop(symbol, None)
    except Exception as e:
        logger.warning(f"[reconcile_positions] failed: {e}")

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    retry_if_exception_type,
    RetryError
)
from ratelimit import limits, sleep_and_retry

import warnings

# ─── A. CONFIGURATION CONSTANTS ─────────────────────────────────────────────────
load_dotenv()
RUN_HEALTH = os.getenv("RUN_HEALTHCHECK", "1") == "1"

# Logging: set root logger to INFO, send to both stderr and a log file
default_log_path = "/var/log/ai-trading-bot.log"
# Use a rotating file handler so the log file cannot grow without bound.
file_handler = logging.handlers.RotatingFileHandler(
    default_log_path, maxBytes=10_000_000, backupCount=5
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),  # stderr for journalctl
        file_handler,
    ],
    force=True,
)
atexit.register(logging.shutdown)
logger = logging.getLogger(__name__)
logging.getLogger("alpaca_trade_api").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# Sentry
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    traces_sample_rate=0.1,
    environment=os.getenv("BOT_MODE", "live"),
)

# Suppress specific pandas_ta warnings
warnings.filterwarnings(
    "ignore",
    message=".*valid feature names.*",
    category=UserWarning
)

# ─── FINBERT SENTIMENT MODEL IMPORTS & FALLBACK ─────────────────────────────────
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _FINBERT_TOKENIZER = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    _FINBERT_MODEL     = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    _FINBERT_MODEL.eval()
    _HUGGINGFACE_AVAILABLE = True
    logger.info("FinBERT loaded successfully")
except Exception as e:
    _HUGGINGFACE_AVAILABLE = False
    _FINBERT_TOKENIZER = None
    _FINBERT_MODEL = None
    logger.warning(f"FinBERT load failed ({e}); falling back to neutral sentiment")

# Prometheus metrics
orders_total            = Counter('bot_orders_total', 'Total orders sent')
order_failures          = Counter('bot_order_failures', 'Order submission failures')
daily_drawdown          = Gauge('bot_daily_drawdown', 'Current daily drawdown fraction')
signals_evaluated       = Counter('bot_signals_evaluated_total', 'Total signals evaluated')
run_all_trades_duration = Histogram('run_all_trades_duration_seconds', 'Time spent in run_all_trades')
minute_cache_hit        = Counter('bot_minute_cache_hits', 'Minute bar cache hits')
minute_cache_miss       = Counter('bot_minute_cache_misses', 'Minute bar cache misses')
daily_cache_hit         = Counter('bot_daily_cache_hits', 'Daily bar cache hits')
daily_cache_miss        = Counter('bot_daily_cache_misses', 'Daily bar cache misses')
event_cooldown_hits     = Counter('bot_event_cooldown_hits', 'Event cooldown hits')
slippage_total          = Counter('bot_slippage_total', 'Cumulative slippage in cents')
slippage_count          = Counter('bot_slippage_count', 'Number of orders with slippage logged')
weekly_drawdown         = Gauge('bot_weekly_drawdown', 'Current weekly drawdown fraction')

# Paths
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
def abspath(fname: str) -> str:
    return os.path.join(BASE_DIR, fname)

TICKERS_FILE        = abspath("tickers.csv")
TRADE_LOG_FILE      = abspath("trades.csv")
SIGNAL_WEIGHTS_FILE = abspath("signal_weights.csv")
EQUITY_FILE         = abspath("last_equity.txt")
PEAK_EQUITY_FILE    = abspath("peak_equity.txt")
HALT_FLAG_PATH      = abspath("halt.flag")
SLIPPAGE_LOG_FILE   = abspath("slippage.csv")

# Hyperparameter files
HYPERPARAMS_FILE     = abspath("hyperparams.json")
BEST_HYPERPARAMS_FILE = abspath("best_hyperparams.json")

def load_hyperparams() -> dict:
    """Load hyperparameters from best_hyperparams.json if present, else default."""
    path = BEST_HYPERPARAMS_FILE if os.path.exists(BEST_HYPERPARAMS_FILE) else HYPERPARAMS_FILE
    if not os.path.exists(path):
        logger.warning(f"Hyperparameter file {path} not found; using defaults")
        return {}
    with open(path) as f:
        return json.load(f)

# <-- NEW: marker file for daily retraining -->
RETRAIN_MARKER_FILE = abspath("last_retrain.txt")

# Main meta‐learner path: this is where retrain.py will dump the new sklearn model each day.
MODEL_PATH          = abspath(os.getenv("MODEL_PATH", "meta_model.pkl"))
MODEL_RF_PATH       = abspath(os.getenv("MODEL_RF_PATH", "model_rf.pkl"))
MODEL_XGB_PATH      = abspath(os.getenv("MODEL_XGB_PATH", "model_xgb.pkl"))
MODEL_LGB_PATH      = abspath(os.getenv("MODEL_LGB_PATH", "model_lgb.pkl"))

REGIME_MODEL_PATH   = abspath("regime_model.pkl")
# (We keep a separate meta‐model for signal‐weight learning, if you use Bayesian/Ridge, etc.)
META_MODEL_PATH     = abspath("meta_model.pkl")

# Strategy mode
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
                "KELLY_FRACTION": 0.6, "CONF_THRESHOLD": 0.75, "CONFIRMATION_COUNT": 2,
                "TAKE_PROFIT_FACTOR": 1.8, "DAILY_LOSS_LIMIT": 0.07,
                "CAPITAL_CAP": 0.08, "TRAILING_FACTOR": 1.2
            }

    def get_config(self) -> dict[str, float]:
        return self.params

BOT_MODE     = os.getenv("BOT_MODE", "balanced")
mode_obj     = BotMode(BOT_MODE)
logger.info(f"Trading mode is set to '{mode_obj.mode}'")
params       = mode_obj.get_config()
params.update(load_hyperparams())

# Other constants
NEWS_API_KEY            = os.getenv("NEWS_API_KEY")
TRAILING_FACTOR         = params.get("TRAILING_FACTOR", 1.2)
SECONDARY_TRAIL_FACTOR  = 1.0
TAKE_PROFIT_FACTOR      = params.get("TAKE_PROFIT_FACTOR", 1.8)
SCALING_FACTOR          = params.get("SCALING_FACTOR", 0.3)
ORDER_TYPE              = 'market'
LIMIT_ORDER_SLIPPAGE    = params.get("LIMIT_ORDER_SLIPPAGE", 0.005)
MAX_POSITION_SIZE       = 1000
SLICE_THRESHOLD         = 50
POV_SLICE_PCT           = params.get("POV_SLICE_PCT", 0.05)
DAILY_LOSS_LIMIT        = params.get("DAILY_LOSS_LIMIT", 0.07)
MAX_PORTFOLIO_POSITIONS = int(os.getenv("MAX_PORTFOLIO_POSITIONS", 15))
CORRELATION_THRESHOLD   = 0.60
SECTOR_EXPOSURE_CAP     = float(os.getenv("SECTOR_EXPOSURE_CAP", "0.4"))
MAX_OPEN_POSITIONS      = int(os.getenv("MAX_OPEN_POSITIONS", "10"))
WEEKLY_DRAWDOWN_LIMIT   = float(os.getenv("WEEKLY_DRAWDOWN_LIMIT", "0.15"))
MARKET_OPEN             = dt_time(6, 30)
MARKET_CLOSE            = dt_time(13, 0)
VOLUME_THRESHOLD        = int(os.getenv("VOLUME_THRESHOLD", "50000"))
ENTRY_START_OFFSET      = timedelta(minutes=params.get("ENTRY_START_OFFSET_MIN", 30))
ENTRY_END_OFFSET        = timedelta(minutes=params.get("ENTRY_END_OFFSET_MIN", 15))
REGIME_LOOKBACK         = 14
REGIME_ATR_THRESHOLD    = 20.0
RF_ESTIMATORS           = 300
RF_MAX_DEPTH            = 3
RF_MIN_SAMPLES_LEAF     = 5
ATR_LENGTH              = 10
CONF_THRESHOLD          = params.get("CONF_THRESHOLD", 0.75)
CONFIRMATION_COUNT      = params.get("CONFIRMATION_COUNT", 2)
CAPITAL_CAP             = params.get("CAPITAL_CAP", 0.08)
DOLLAR_RISK_LIMIT       = float(os.getenv("DOLLAR_RISK_LIMIT", "0.02"))
PACIFIC                 = ZoneInfo("America/Los_Angeles")
PDT_DAY_TRADE_LIMIT     = params.get("PDT_DAY_TRADE_LIMIT", 3)
PDT_EQUITY_THRESHOLD    = params.get("PDT_EQUITY_THRESHOLD", 25_000.0)
BUY_THRESHOLD           = params.get("BUY_THRESHOLD", 0.2)
FINNHUB_RPM             = int(os.getenv("FINNHUB_RPM", "60"))

# Regime symbols (makes SPY configurable)
REGIME_SYMBOLS = ["SPY"]

# ─── THREAD-SAFETY LOCKS & CIRCUIT BREAKER ─────────────────────────────────────
cache_lock      = Lock()
targets_lock    = Lock()
vol_lock        = Lock()
sentiment_lock  = Lock()
slippage_lock   = Lock()
meta_lock       = Lock()

breaker = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60)
executor = ThreadPoolExecutor(max_workers=4)

# EVENT cooldown
_LAST_EVENT_TS = {}
EVENT_COOLDOWN = 15.0  # seconds

# Loss streak kill-switch
_LOSS_STREAK = 0
_STREAK_HALT_UNTIL: Optional[datetime] = None

# Volatility stats (for SPY ATR mean/std)
_VOL_STATS = {"mean": None, "std": None, "last_update": None}

# Slippage logs (in-memory for quick access)
_slippage_log: List[Tuple[str, float, float, datetime]] = []  # (symbol, expected, actual, timestamp)
# Ensure persistent slippage log file exists
if not os.path.exists(SLIPPAGE_LOG_FILE):
    with open(SLIPPAGE_LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "symbol", "expected", "actual", "slippage_cents"])

# Sector cache for portfolio exposure calculations
_SECTOR_CACHE: Dict[str, str] = {}

# Weekly drawdown tracking
week_start_equity: Optional[Tuple[date, float]] = None

# ─── TYPED EXCEPTION ─────────────────────────────────────────────────────────
class DataFetchError(Exception):
    pass

# ─── B. CLIENTS & SINGLETONS ─────────────────────────────────────────────────
finnhub_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))

def ensure_alpaca_credentials() -> None:
    if not os.getenv("APCA_API_KEY_ID") or not os.getenv("APCA_API_SECRET_KEY"):
        raise RuntimeError("Missing Alpaca API credentials; please check .env")
ensure_alpaca_credentials()

# Prometheus-safe account fetch
@breaker
def safe_alpaca_get_account(ctx: 'BotContext'):
    return ctx.api.get_account()

# ─── C. HELPERS ────────────────────────────────────────────────────────────────
def chunked(iterable: Sequence, n: int):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

def ttl_seconds() -> int:
    """Configurable TTL for minute-bar cache (default 60s)."""
    return int(os.getenv("MINUTE_CACHE_TTL", "60"))

def compute_spy_vol_stats(ctx: 'BotContext') -> None:
    """Compute daily ATR mean/std on SPY for the past 1 year."""
    global _VOL_STATS
    today = date.today()
    with vol_lock:
        if _VOL_STATS["last_update"] == today:
            return

    df = ctx.data_fetcher.get_daily_df(ctx, REGIME_SYMBOLS[0])
    if df is None or len(df) < 252 + ATR_LENGTH:
        return

    # Compute ATR series for last 252 trading days
    atr_series = ta.atr(df["High"], df["Low"], df["Close"], length=ATR_LENGTH).dropna()
    if len(atr_series) < 252:
        return

    recent = atr_series.iloc[-252:]
    mean_val = float(recent.mean())
    std_val = float(recent.std())

    with vol_lock:
        _VOL_STATS["mean"] = mean_val
        _VOL_STATS["std"] = std_val
        _VOL_STATS["last_update"] = today

    logger.info("SPY_VOL_STATS_UPDATED", extra={"mean": mean_val, "std": std_val})

def is_high_vol_thr_spy() -> bool:
    """Return True if SPY ATR > mean + 2*std."""
    with vol_lock:
        mean = _VOL_STATS["mean"]
        std = _VOL_STATS["std"]
    if mean is None or std is None:
        return False

    with cache_lock:
        spy_df = data_fetcher._daily_cache.get(REGIME_SYMBOLS[0])
    if spy_df is None or len(spy_df) < ATR_LENGTH:
        return False

    atr_series = ta.atr(spy_df["High"], spy_df["Low"], spy_df["Close"], length=ATR_LENGTH)
    if atr_series.empty:
        return False

    current_atr = float(atr_series.iloc[-1])
    return (current_atr - mean) / std >= 2

def is_high_vol_regime() -> bool:
    """
    Wrapper for is_high_vol_thr_spy to be used inside update_trailing_stop and execute_entry.
    Returns True if SPY is in a high-volatility regime (ATR > mean + 2*std).
    """
    return is_high_vol_thr_spy()

# ─── D. DATA FETCHERS ─────────────────────────────────────────────────────────
class FinnhubFetcher:
    def __init__(self, calls_per_minute: int = FINNHUB_RPM):
        self.max_calls = calls_per_minute
        self._timestamps = deque()
        self.client = finnhub_client

    def _throttle(self):
        while True:
            now_ts = pytime.time()
            # drop timestamps older than 60 seconds
            while self._timestamps and now_ts - self._timestamps[0] > 60:
                self._timestamps.popleft()
            if len(self._timestamps) < self.max_calls:
                self._timestamps.append(now_ts)
                return
            wait_secs = 60 - (now_ts - self._timestamps[0]) + random.uniform(0.1, 0.5)
            logger.debug(f"[FH] rate-limit reached; sleeping {wait_secs:.2f}s")
            pytime.sleep(wait_secs)

    def _parse_period(self, period: str) -> int:
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
                logger.warning(f"[FH] no data for {sym}: status={resp.get('s')}")
                frames.append(pd.DataFrame())
                continue
            df = pd.DataFrame({
                'Open':   resp['o'],
                'High':   resp['h'],
                'Low':    resp['l'],
                'Close':  resp['c'],
                'Volume': resp['v'],
            }, index=pd.to_datetime(resp['t'], unit='s', utc=True))
            df.index = df.index.tz_convert(None)
            frames.append(df)

        if not frames:
            return pd.DataFrame()
        if len(frames) == 1:
            return frames[0]
        return pd.concat(frames, axis=1, keys=syms, names=['Symbol','Field'])

# Instantiate FinnhubFetcher singleton
fh = FinnhubFetcher()

_last_fh_prefetch_date: Optional[date] = None

@dataclass
class DataFetcher:
    def __post_init__(self):
        self._daily_cache: dict[str, Optional[pd.DataFrame]] = {}
        self._minute_cache: dict[str, Optional[pd.DataFrame]] = {}
        self._minute_timestamps: dict[str, datetime] = {}

    def get_daily_df(self, ctx: 'BotContext', symbol: str) -> Optional[pd.DataFrame]:
        # Regime-based fetch symbols
        with cache_lock:
            if symbol in self._daily_cache and symbol in REGIME_SYMBOLS:
                daily_cache_hit.inc()
                return self._daily_cache[symbol]

        if symbol in REGIME_SYMBOLS:
            today_date = date.today()
            start = (today_date - timedelta(days=365)).isoformat()
            end = today_date.isoformat()
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
                "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"
            })
            with cache_lock:
                self._daily_cache[symbol] = df
            daily_cache_hit.inc()
            return df

        with cache_lock:
            if symbol in self._daily_cache:
                daily_cache_hit.inc()
                return self._daily_cache[symbol]

        try:
            df = fh.fetch(symbol, period="1mo", interval="1d")
            if df is None or df.empty:
                raise DataFetchError(f"No daily data for {symbol}")
            daily_cache_hit.inc()
        except Exception as e:
            logger.warning(
                f"[DataFetcher] daily fetch failed for {symbol}: {e}")
            # Fallback to yfinance if Finnhub fails
            try:
                df_yf = yf.download(symbol, period="1mo", interval="1d", progress=False)
                if not df_yf.empty:
                    df_yf.index = pd.to_datetime(df_yf.index).tz_localize(None)
                    df = df_yf.rename(columns=lambda c: c.title())[
                        ["Open", "High", "Low", "Close", "Volume"]
                    ]
                    logger.warning(
                        f"[DataFetcher] fallback to yfinance daily for {symbol}")
                else:
                    df = None
            except Exception as e2:
                logger.error(
                    f"[DataFetcher] yfinance daily fallback failed for {symbol}: {e2}")
                df = None
            daily_cache_miss.inc()

        with cache_lock:
            self._daily_cache[symbol] = df
        return df

    def get_minute_df(self, ctx: 'BotContext', symbol: str) -> Optional[pd.DataFrame]:
        now_utc = datetime.now(timezone.utc)
        with cache_lock:
            last_ts = self._minute_timestamps.get(symbol)
            if last_ts and last_ts > now_utc - timedelta(seconds=ttl_seconds()):
                minute_cache_hit.inc()
                return self._minute_cache[symbol]

        minute_cache_miss.inc()
        df: Optional[pd.DataFrame] = None

        # 1) Alpaca IEX fetch
        try:
            bars = ctx.api.get_bars(
                symbol,
                TimeFrame.Minute,
                limit=390 * 5,
                feed="iex"
            ).df
            if not bars.empty:
                bars = bars.rename(columns={
                    "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"
                })
                if "symbol" in bars.columns:
                    bars = bars.drop(columns=["symbol"])
                bars.index = pd.to_datetime(bars.index).tz_localize(None)
                df = bars[["Open", "High", "Low", "Close", "Volume"]]
                logger.debug(f"[DataFetcher] minute bars via Alpaca IEX for {symbol}")
            else:
                # If empty, fall back to Finnhub 1-min
                df_fh = fh.fetch(symbol, period="5d", interval="1m")
                if df_fh is not None and not df_fh.empty:
                    df = df_fh.rename(columns=lambda c: c if c in ["Open", "High", "Low", "Close", "Volume"] else c)
                    logger.warning(f"[DataFetcher] fallback to Finnhub 1-min for {symbol}")
        except Exception as e:
            logger.warning(f"[DataFetcher] Alpaca minute fetch failed for {symbol}: {e}")
            # fallback to Finnhub if Alpaca fails
            try:
                df_fh = fh.fetch(symbol, period="5d", interval="1m")
                if df_fh is not None and not df_fh.empty:
                    df = df_fh.rename(columns=lambda c: c if c in ["Open", "High", "Low", "Close", "Volume"] else c)
                    logger.warning(f"[DataFetcher] fallback to Finnhub 1-min for {symbol}")
                else:
                    df = None
            except Exception:
                df = None
            # Fallback to yfinance if Finnhub fails too
            if df is None:
                try:
                    df_yf = yf.download(symbol, period="5d", interval="1m", progress=False)
                    if not df_yf.empty:
                        df_yf.index = pd.to_datetime(df_yf.index).tz_localize(None)
                        df = df_yf.rename(columns=lambda c: c.title())[["Open", "High", "Low", "Close", "Volume"]]
                        logger.warning(f"[DataFetcher] fallback to yfinance 1-min for {symbol}")
                except Exception as e2:
                    logger.error(f"[DataFetcher] yfinance minute fallback failed for {symbol}: {e2}")

        with cache_lock:
            self._minute_cache[symbol] = df
            self._minute_timestamps[symbol] = now_utc
        return df

    def get_historical_minute(
        self,
        ctx: 'BotContext',           # ← still needs ctx here, per retrain.py
        symbol: str,
        start_date: date,
        end_date: date
    ) -> Optional[pd.DataFrame]:
        """
        Fetch all minute bars for `symbol` between start_date and end_date (inclusive),
        by calling Alpaca’s get_bars for each calendar day. Returns a DataFrame
        indexed by naive Timestamps, or None if no data was returned at all.
        """
        all_days: list[pd.DataFrame] = []
        current_day = start_date

        while current_day <= end_date:
            # Build ISO strings for that single day:
            day_start_iso = f"{current_day.isoformat()}T00:00:00Z"
            day_end_iso   = f"{current_day.isoformat()}T23:59:59Z"

            try:
                bars_day = ctx.api.get_bars(
                    symbol,
                    TimeFrame.Minute,
                    start=day_start_iso,
                    end=day_end_iso,
                    limit=10000
                ).df
            except Exception:
                bars_day = None

            if bars_day is not None and not bars_day.empty:
                # Drop "symbol" column if present, rename to Title-case, drop tz, keep only OHLCV
                if "symbol" in bars_day.columns:
                    bars_day = bars_day.drop(columns=["symbol"], errors="ignore")

                bars_day.index = pd.to_datetime(bars_day.index).tz_localize(None)
                bars_day = bars_day.rename(columns={
                    "open":   "Open",
                    "high":   "High",
                    "low":    "Low",
                    "close":  "Close",
                    "volume": "Volume",
                })
                bars_day = bars_day[["Open", "High", "Low", "Close", "Volume"]]
                all_days.append(bars_day)

            current_day += timedelta(days=1)

        if not all_days:
            return None

        combined = pd.concat(all_days, axis=0)
        combined = combined[~combined.index.duplicated(keep="first")]
        combined = combined.sort_index()
        return combined
        
# ─── E. TRADE LOGGER ───────────────────────────────────────────────────────────
class TradeLogger:
    def __init__(self, path: str = TRADE_LOG_FILE) -> None:
        self.path = path
        if not os.path.exists(path):
            with portalocker.Lock(path, 'w', timeout=5) as f:
                csv.writer(f).writerow([
                    "symbol","entry_time","entry_price",
                    "exit_time","exit_price","qty","side",
                    "strategy","classification","signal_tags"
                ])

    def log_entry(self, symbol: str, price: float, qty: int, side: str, strategy: str, signal_tags: str="") -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with portalocker.Lock(self.path, 'a', timeout=5) as f:
            csv.writer(f).writerow([symbol, now_iso, price, "","", qty, side, strategy, "", signal_tags])

    def log_exit(self, symbol: str, exit_price: float) -> None:
        global _LOSS_STREAK, _STREAK_HALT_UNTIL
        with portalocker.Lock(self.path, 'r+', timeout=5) as f:
            rows = list(csv.reader(f))
            header, data = rows[0], rows[1:]
            pnl = 0.0
            for row in data:
                if row[0] == symbol and row[3] == "":
                    entry_t = datetime.fromisoformat(row[1])
                    days = (datetime.now(timezone.utc) - entry_t).days
                    cls = ("day_trade" if days == 0
                        else "swing_trade" if days < 5
                        else "long_trade")
                    row[3], row[4], row[8] = datetime.now(timezone.utc).isoformat(), exit_price, cls
                    # Compute PnL
                    entry_price = float(row[2])
                    pnl = (exit_price - entry_price) * (1 if row[6] == "buy" else -1)
                    break
            f.seek(0); f.truncate()
            w = csv.writer(f)
            w.writerow(header); w.writerows(data)

        # Update streak-based kill-switch
        if pnl < 0:
            _LOSS_STREAK += 1
        else:
            _LOSS_STREAK = 0
        if _LOSS_STREAK >= 3:
            _STREAK_HALT_UNTIL = datetime.now(PACIFIC) + timedelta(minutes=60)
            logger.warning("STREAK_HALT_TRIGGERED", extra={"loss_streak": _LOSS_STREAK, "halt_until": _STREAK_HALT_UNTIL})

def _parse_local_positions() -> Dict[str, int]:
    """Return current local open positions from the trade log."""
    positions: Dict[str, int] = {}
    if not os.path.exists(TRADE_LOG_FILE):
        return positions
    df = pd.read_csv(TRADE_LOG_FILE)
    for _, row in df.iterrows():
        if str(row.get("exit_time", "")) != "":
            continue
        qty = int(row.qty)
        qty = qty if row.side == "buy" else -qty
        positions[row.symbol] = positions.get(row.symbol, 0) + qty
    positions = {k: v for k, v in positions.items() if v != 0}
    return positions

def audit_positions(ctx: "BotContext") -> None:
    """
    Fetch local vs. broker positions and submit market orders to correct any mismatch.
    """
    # 1) Read local open positions from the trade log
    local = _parse_local_positions()

    # 2) Fetch remote (broker) positions
    try:
        remote = {p.symbol: int(p.qty) for p in ctx.api.list_positions()}
    except Exception:
        return

    # 3) For any symbol in remote whose remote_qty != local_qty, correct via market order
    for sym, rq in remote.items():
        lq = local.get(sym, 0)
        if lq != rq:
            diff = rq - lq
            if diff > 0:
                # Broker has more shares than local: sell off the excess
                try:
                    ctx.api.submit_order(symbol=sym, qty=diff, side="sell", type="market")
                except Exception:
                    pass
            else:
                # Broker has fewer shares than local: buy back the missing shares
                try:
                    ctx.api.submit_order(symbol=sym, qty=abs(diff), side="buy", type="market")
                except Exception:
                    pass

    # 4) For any symbol in local that is not in remote, submit order matching the local side
    for sym, lq in local.items():
        if sym not in remote:
            try:
                side = "buy" if lq > 0 else "sell"
                ctx.api.submit_order(symbol=sym, qty=abs(lq), side=side, type="market")
            except Exception:
                pass

def validate_open_orders(ctx: "BotContext") -> None:
    try:
        open_orders = ctx.api.list_orders(status="open")
    except Exception:
        return

    now = datetime.now(timezone.utc)
    for od in open_orders:
        created = pd.to_datetime(getattr(od, "created_at", now))
        age = (now - created).total_seconds() / 60.0

        if age > 5 and getattr(od, "status", "").lower() in {"new", "accepted"}:
            try:
                ctx.api.cancel_order(od.id)
                qty = int(getattr(od, "qty", 0))
                side = getattr(od, "side", "")
                if qty > 0 and side in {"buy", "sell"}:
                    # Resubmit exactly the same size and side as a market order
                    ctx.api.submit_order(symbol=od.symbol, qty=qty, side=side, type="market")
            except Exception:
                pass

    # After canceling/replacing any stuck orders, fix any position mismatches
    audit_positions(ctx)

# ─── F. SIGNAL MANAGER & HELPER FUNCTIONS ─────────────────────────────────────
_LAST_PRICE: Dict[str, float] = {}
_SENTIMENT_CACHE: Dict[str, Tuple[float, float]] = {}  # {ticker: (timestamp, score)}
PRICE_TTL_PCT = 0.005  # only fetch sentiment if price moved > 0.5%
SENTIMENT_TTL_SEC = 600  # 10 minutes

class SignalManager:
    def __init__(self) -> None:
        self.momentum_lookback = 5
        self.mean_rev_lookback = 20
        self.mean_rev_zscore_threshold = 2.0
        self.regime_volatility_threshold = REGIME_ATR_THRESHOLD
        self.last_components: List[Tuple[int, float, str]] = []

    def signal_momentum(self, df: pd.DataFrame, model=None) -> Tuple[int, float, str]:
        if df is None or len(df) <= self.momentum_lookback:
            return -1, 0.0, 'momentum'
        try:
            df['momentum'] = df['Close'].pct_change(self.momentum_lookback)
            val = df['momentum'].iloc[-1]
            s = 1 if val > 0 else -1 if val < 0 else -1
            w = min(abs(val) * 10, 1.0)
            return s, w, 'momentum'
        except Exception:
            logger.exception("Error in signal_momentum")
            return -1, 0.0, 'momentum'

    def signal_mean_reversion(self, df: pd.DataFrame, model=None) -> Tuple[int, float, str]:
        if df is None or len(df) < self.mean_rev_lookback:
            return -1, 0.0, 'mean_reversion'
        try:
            ma = df['Close'].rolling(self.mean_rev_lookback).mean()
            sd = df['Close'].rolling(self.mean_rev_lookback).std()
            df['zscore'] = (df['Close'] - ma) / sd
            val = df['zscore'].iloc[-1]
            s = -1 if val > self.mean_rev_zscore_threshold else 1 if val < -self.mean_rev_zscore_threshold else -1
            w = min(abs(val) / 3, 1.0)
            return s, w, 'mean_reversion'
        except Exception:
            logger.exception("Error in signal_mean_reversion")
            return -1, 0.0, 'mean_reversion'

    def signal_stochrsi(self, df: pd.DataFrame, model=None) -> Tuple[int, float, str]:
        if df is None or 'stochrsi' not in df or df['stochrsi'].dropna().empty:
            return -1, 0.0, 'stochrsi'
        try:
            val = df['stochrsi'].iloc[-1]
            s = 1 if val < 0.2 else -1 if val > 0.8 else -1
            return s, 0.3, 'stochrsi'
        except Exception:
            logger.exception("Error in signal_stochrsi")
            return -1, 0.0, 'stochrsi'

    def signal_obv(self, df: pd.DataFrame, model=None) -> Tuple[int, float, str]:
        if df is None or len(df) < 6:
            return -1, 0.0, 'obv'
        try:
            obv = pd.Series(ta.obv(df['Close'], df['Volume']).values)
            if len(obv) < 5:
                return -1, 0.0, 'obv'
            slope = np.polyfit(range(5), obv.tail(5), 1)[0]
            s = 1 if slope > 0 else -1 if slope < 0 else -1
            w = min(abs(slope) / 1e6, 1.0)
            return s, w, 'obv'
        except Exception:
            logger.exception("Error in signal_obv")
            return -1, 0.0, 'obv'

    def signal_vsa(self, df: pd.DataFrame, model=None) -> Tuple[int, float, str]:
        if df is None or len(df) < 20:
            return -1, 0.0, 'vsa'
        try:
            body = abs(df['Close'] - df['Open'])
            vsa = df['Volume'] * body
            score = vsa.iloc[-1]
            avg = vsa.rolling(20).mean().iloc[-1]
            s = 1 if df['Close'].iloc[-1] > df['Open'].iloc[-1] else -1 if df['Close'].iloc[-1] < df['Open'].iloc[-1] else -1
            w = min(score / avg, 1.0)
            return s, w, 'vsa'
        except Exception:
            logger.exception("Error in signal_vsa")
            return -1, 0.0, 'vsa'

    def signal_ml(self, df: pd.DataFrame, model: Any=None) -> Tuple[int, float, str]:
        """Machine learning prediction signal with probability logging."""
        try:
            if hasattr(model, "feature_names_in_"):
                feat = list(model.feature_names_in_)
            else:
                feat = ['rsi', 'macd', 'atr', 'vwap', 'sma_50', 'sma_200']

            X = df[feat].iloc[-1].values.reshape(1, -1)
            pred = model.predict(X)[0]
            proba = float(model.predict_proba(X)[0][pred])
            s = 1 if pred == 1 else -1
            logger.info(
                "ML_SIGNAL",
                extra={"prediction": int(pred), "probability": proba}
            )
            return s, proba, 'ml'
        except Exception as e:
            logger.exception(f"signal_ml failed: {e}")
            return -1, 0.0, 'ml'

    def signal_sentiment(self, ctx: 'BotContext', ticker: str, df: pd.DataFrame=None, model: Any=None) -> Tuple[int, float, str]:
        """
        Only fetch sentiment if price has moved > PRICE_TTL_PCT; otherwise, return cached/neutral.
        """
        if df is None or df.empty:
            return -1, 0.0, "sentiment"

        latest_close = float(df["Close"].iloc[-1])
        with sentiment_lock:
            prev_close = _LAST_PRICE.get(ticker, None)

        # If price hasn’t moved enough, return cached or neutral
        if prev_close is not None and abs(latest_close - prev_close) / prev_close < PRICE_TTL_PCT:
            with sentiment_lock:
                cached = _SENTIMENT_CACHE.get(ticker)
                if cached and (pytime.time() - cached[0] < SENTIMENT_TTL_SEC):
                    score = cached[1]
                else:
                    score = 0.0
                    _SENTIMENT_CACHE[ticker] = (pytime.time(), score)
        else:
            # Price moved enough → fetch fresh sentiment
            try:
                score = fetch_sentiment(ctx, ticker)
            except Exception as e:
                logger.warning(f"[signal_sentiment] {ticker} error: {e}")
                score = 0.0

        # Update last‐seen price & cache
        with sentiment_lock:
            _LAST_PRICE[ticker] = latest_close
            _SENTIMENT_CACHE[ticker] = (pytime.time(), score)

        score = max(-1.0, min(1.0, score))
        s = 1 if score > 0 else -1 if score < 0 else -1
        weight = abs(score)
        if is_high_vol_regime():
            weight *= 1.5
        return s, weight, 'sentiment'

    def signal_regime(self, df: pd.DataFrame, model=None) -> Tuple[int, float, str]:
        ok = check_market_regime()
        s = 1 if ok else -1
        return s, 1.0, 'regime'

    def load_signal_weights(self) -> dict[str, float]:
        if not os.path.exists(SIGNAL_WEIGHTS_FILE):
            return {}
        df = pd.read_csv(SIGNAL_WEIGHTS_FILE)
        return {row['signal']: row['weight'] for _, row in df.iterrows()}

    def evaluate(self, ctx: 'BotContext', df: pd.DataFrame, ticker: str, model: Any) -> Tuple[int, float, str]:
        """
        Evaluate all signals, allowing short (s=-1) and long (s=1) components.
        Returns (final_signal, confidence ∈ [0,1], concatenated labels).
        """
        signals: List[Tuple[int, float, str]] = []
        allowed_tags = set(load_global_signal_performance() or [])
        weights = self.load_signal_weights()

        # Track total signals evaluated
        signals_evaluated.inc()

        fns = [
            self.signal_momentum,
            self.signal_mean_reversion,
            self.signal_ml,
            self.signal_sentiment,
            self.signal_regime,
            self.signal_stochrsi,
            self.signal_obv,
            self.signal_vsa,
        ]

        for fn in fns:
            try:
                if fn == self.signal_sentiment:
                    s, w, lab = fn(ctx, ticker, df, model)
                else:
                    s, w, lab = fn(df, model)
                if allowed_tags and lab not in allowed_tags:
                    continue
                if s in (-1, 1):
                    signals.append((s, weights.get(lab, w), lab))
            except Exception:
                continue

        if not signals:
            return -1, 0.0, 'no_signal'

        self.last_components = signals

        ml_prob = next((w for s, w, l in signals if l == 'ml'), 0.5)
        adj = []
        for s, w, l in signals:
            if l != 'ml':
                adj.append((s, w * ml_prob, l))
            else:
                adj.append((s, w, l))

        score = sum(s * w for s, w, _ in adj)
        conf = min(abs(score), 1.0)
        if score > 0.5:
            final = 1
        elif score < -0.5:
            final = -1
        else:
            final = -1
        label = "+".join(lab for _, _, lab in adj)
        return final, conf, label

# ─── G. BOT CONTEXT ───────────────────────────────────────────────────────────
@dataclass
class BotContext:
    api: REST
    data_fetcher: DataFetcher
    signal_manager: SignalManager
    trade_logger: TradeLogger
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
    stop_targets: dict[str, float] = field(default_factory=dict)
    portfolio_weights: dict[str, float] = field(default_factory=dict)

data_fetcher   = DataFetcher()
signal_manager = SignalManager()
trade_logger   = TradeLogger()
ctx = BotContext(
    api=REST(
        os.getenv("APCA_API_KEY_ID"),
        os.getenv("APCA_API_SECRET_KEY"),
        base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    ),
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
    kelly_fraction=params.get("KELLY_FRACTION", 0.6),
    confirmation_count={},
    trailing_extremes={},
    take_profit_targets={},
    stop_targets={},
    portfolio_weights={},
)

# Warm up regime history cache so initial regime checks pass
try:
    ctx.data_fetcher.get_daily_df(ctx, REGIME_SYMBOLS[0])
except Exception as e:
    logger.warning(f"[warm_cache] failed to seed regime history: {e}")

# ─── H. MARKET HOURS GUARD ────────────────────────────────────────────────────
nyse = mcal.get_calendar("XNYS")
def in_trading_hours(ts: pd.Timestamp) -> bool:
    schedule_today = nyse.schedule(start_date=ts.date(), end_date=ts.date())
    if schedule_today.empty:
        return False
    return schedule_today.market_open.iloc[0] <= ts <= schedule_today.market_close.iloc[0]

# ─── I. SENTIMENT & EVENTS ────────────────────────────────────────────────────
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

    try:
        soup = BeautifulSoup(r.content, "lxml")
        texts = []
        for a in soup.find_all("a", string=re.compile(r"8[- ]?K")):
            tr = a.find_parent("tr")
            tds = tr.find_all("td") if tr else []
            if len(tds) >= 4:
                texts.append(tds[-1].get_text(strip=True))
        return " ".join(texts)
    except Exception as e:
        logger.warning(f"[get_sec_headlines] parse failed for {ticker}: {e}")
        return ""

@sleep_and_retry
@limits(calls=60, period=60)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type((requests.RequestException, DataFetchError))
)
def fetch_sentiment(ctx: BotContext, ticker: str) -> float:
    """
    Fetch sentiment via NewsAPI + FinBERT + Form 4 signal.
    Uses a simple in-memory TTL cache to avoid hitting NewsAPI too often.
    If FinBERT isn’t available, return neutral 0.0.
    """
    if not NEWS_API_KEY:
        return 0.0

    now_ts = pytime.time()
    with sentiment_lock:
        cached = _SENTIMENT_CACHE.get(ticker)
        if cached:
            last_ts, last_score = cached
            if now_ts - last_ts < SENTIMENT_TTL_SEC:
                return last_score

    # Cache miss or stale → fetch fresh
    # 1) Fetch NewsAPI articles
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
            with sentiment_lock:
                _SENTIMENT_CACHE[ticker] = (now_ts, 0.0)
            return 0.0
        raise

    payload = resp.json()
    articles = payload.get("articles", [])
    scores = []
    if articles:
        for art in articles:
            text = (art.get("title") or "") + ". " + (art.get("description") or "")
            if text.strip():
                scores.append(predict_text_sentiment(text))
    news_score = float(sum(scores) / len(scores)) if scores else 0.0

    # 2) Fetch Form 4 data (insider trades)
    form4_score = 0.0
    try:
        form4 = fetch_form4_filings(ticker)
        # If any insider buy in last 7 days > $50k, boost sentiment
        for filing in form4:
            if filing["type"] == "buy" and filing["dollar_amount"] > 50_000:
                form4_score += 0.1
    except Exception as e:
        logger.warning(f"[fetch_sentiment] Form4 fetch failed for {ticker}: {e}")

    final_score = 0.8 * news_score + 0.2 * form4_score
    final_score = max(-1.0, min(1.0, final_score))
    with sentiment_lock:
        _SENTIMENT_CACHE[ticker] = (now_ts, final_score)
    return final_score

def predict_text_sentiment(text: str) -> float:
    """
    Uses FinBERT (if available) to assign a sentiment score ∈ [–1, +1].
    If FinBERT is unavailable, return 0.0.
    """
    if _HUGGINGFACE_AVAILABLE and _FINBERT_MODEL and _FINBERT_TOKENIZER:
        try:
            inputs = _FINBERT_TOKENIZER(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            )
            with torch.no_grad():
                outputs = _FINBERT_MODEL(**inputs)
                logits = outputs.logits[0]  # shape = (3,)
                probs = torch.softmax(logits, dim=0)  # [p_neg, p_neu, p_pos]

            neg, neu, pos = probs.tolist()
            return float(pos - neg)
        except Exception as e:
            logger.warning(f"[predict_text_sentiment] FinBERT inference failed ({e}); returning neutral")
    return 0.0

def fetch_form4_filings(ticker: str) -> List[dict]:
    """
    Scrape SEC Form 4 filings for insider trade info.
    Returns a list of dicts: {"date": datetime, "type": "buy"/"sell", "dollar_amount": float}.
    """
    url = f"https://www.sec.gov/cgi-bin/own-disp?action=getowner&CIK={ticker}&type=4"
    r = requests.get(url, headers={"User-Agent": "AI Trading Bot"}, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "lxml")
    filings = []
    # Parse table rows (approximate)
    table = soup.find("table", {"class": "tableFile2"})
    if not table:
        return filings
    rows = table.find_all("tr")[1:]  # skip header
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 6:
            continue
        date_str = cols[3].get_text(strip=True)
        try:
            fdate = datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            continue
        txn_type = cols[4].get_text(strip=True).lower()  # "purchase" or "sale"
        amt_str = cols[5].get_text(strip=True).replace("$", "").replace(",", "")
        try:
            amt = float(amt_str)
        except Exception:
            amt = 0.0
        filings.append({"date": fdate, "type": ("buy" if "purchase" in txn_type else "sell"), "dollar_amount": amt})
    return filings

def _can_fetch_events(symbol: str) -> bool:
    now_ts = pytime.time()
    last_ts = _LAST_EVENT_TS.get(symbol, 0)
    if now_ts - last_ts < EVENT_COOLDOWN:
        event_cooldown_hits.inc()
        return False
    _LAST_EVENT_TS[symbol] = now_ts
    return True

_calendar_cache: Dict[str, pd.DataFrame] = {}
_calendar_last_fetch: Dict[str, date] = {}
def get_calendar_safe(symbol: str) -> pd.DataFrame:
    today_date = date.today()
    if symbol in _calendar_cache and _calendar_last_fetch.get(symbol) == today_date:
        return _calendar_cache[symbol]
    try:
        cal = yf.Ticker(symbol).calendar
    except YFRateLimitError:
        logger.warning(f"[Events] Rate limited for {symbol}; skipping events.")
        cal = pd.DataFrame()
    except Exception as e:
        logger.error(f"[Events] Error fetching calendar for {symbol}: {e}")
        cal = pd.DataFrame()
    _calendar_cache[symbol] = cal
    _calendar_last_fetch[symbol] = today_date
    return cal

def is_near_event(symbol: str, days: int = 3) -> bool:
    cal = get_calendar_safe(symbol)
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
    today_ts = pd.Timestamp.now().normalize()
    cutoff = today_ts + pd.Timedelta(days=days)
    return any(today_ts <= d <= cutoff for d in dates)

# ─── J. RISK & GUARDS ─────────────────────────────────────────────────────────
day_start_equity: Optional[Tuple[date, float]] = None
last_drawdown: float = 0.0

@sleep_and_retry
@limits(calls=200, period=60)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type(APIError)
)
def check_daily_loss(ctx: BotContext) -> bool:
    global day_start_equity, last_drawdown
    acct = safe_alpaca_get_account(ctx)
    equity = float(acct.equity)
    today_date = date.today()
    limit = params.get("DAILY_LOSS_LIMIT", 0.07)

    if day_start_equity is None or day_start_equity[0] != today_date:
        if last_drawdown >= 0.05:
            limit = 0.03
        last_drawdown = (
            (day_start_equity[1] - equity) / day_start_equity[1]
            if day_start_equity else 0.0
        )
        day_start_equity = (today_date, equity)
        daily_drawdown.set(0.0)
        return False

    loss = (day_start_equity[1] - equity) / day_start_equity[1]
    daily_drawdown.set(loss)
    if loss > 0.05:
        sentry_sdk.capture_message(f"[WARNING] Daily drawdown = {loss:.2%}")
    return loss >= limit

def check_weekly_loss(ctx: BotContext) -> bool:
    """Weekly portfolio drawdown guard."""
    global week_start_equity
    acct = safe_alpaca_get_account(ctx)
    equity = float(acct.equity)
    today_date = date.today()
    week_start = today_date - timedelta(days=today_date.weekday())

    if week_start_equity is None or week_start_equity[0] != week_start:
        week_start_equity = (week_start, equity)
        weekly_drawdown.set(0.0)
        return False

    loss = (week_start_equity[1] - equity) / week_start_equity[1]
    weekly_drawdown.set(loss)
    return loss >= WEEKLY_DRAWDOWN_LIMIT

def count_day_trades() -> int:
    if not os.path.exists(TRADE_LOG_FILE):
        return 0
    df = pd.read_csv(TRADE_LOG_FILE)
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
    df = df.dropna(subset=["entry_time", "exit_time"])
    today_ts = pd.Timestamp.now().normalize()
    bdays = pd.bdate_range(end=today_ts, periods=5)
    df["entry_date"] = df["entry_time"].dt.normalize()
    df["exit_date"] = df["exit_time"].dt.normalize()
    mask = (
        (df["entry_date"].isin(bdays)) &
        (df["exit_date"].isin(bdays)) &
        (df["entry_date"] == df["exit_date"])
    )
    return int(mask.sum())

_running = False

@sleep_and_retry
@limits(calls=200, period=60)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type(APIError)
)
def check_pdt_rule(ctx: BotContext) -> bool:
    acct = safe_alpaca_get_account(ctx)
    equity = float(acct.equity)

    api_day_trades = (
        getattr(acct, "pattern_day_trades", None)
        or getattr(acct, "pattern_day_trades_count", None)
    )
    api_buying_pw = (
        getattr(acct, "daytrade_buying_power", None)
        or getattr(acct, "day_trade_buying_power", None)
    )

    logger.info(
        "PDT_CHECK",
        extra={
            "equity": equity,
            "api_day_trades": api_day_trades,
            "api_buying_pw": api_buying_pw,
        },
    )

    if api_day_trades is not None and api_day_trades >= PDT_DAY_TRADE_LIMIT:
        logger.info("SKIP_PDT_RULE", extra={"api_day_trades": api_day_trades})
        return True

    if equity < PDT_EQUITY_THRESHOLD:
        if api_buying_pw and float(api_buying_pw) > 0:
            logger.warning(
                "PDT_EQUITY_LOW", extra={"equity": equity, "buying_pw": api_buying_pw}
            )
        else:
            logger.warning(
                "PDT_EQUITY_LOW_NO_BP",
                extra={"equity": equity, "buying_pw": api_buying_pw},
            )
            return True

    return False

def check_halt_flag() -> bool:
    if not os.path.exists(HALT_FLAG_PATH):
        return False
    mtime = os.path.getmtime(HALT_FLAG_PATH)
    age = datetime.now(timezone.utc) - datetime.fromtimestamp(mtime, timezone.utc)
    return age < timedelta(hours=1)

def too_many_positions(ctx: BotContext) -> bool:
    try:
        return len(ctx.api.list_positions()) >= MAX_PORTFOLIO_POSITIONS
    except Exception:
        logger.warning("[too_many_positions] Could not fetch positions")
        return False

def too_correlated(sym: str) -> bool:
    if not os.path.exists(TRADE_LOG_FILE):
        return False
    df = pd.read_csv(TRADE_LOG_FILE)
    open_syms = df.loc[df.exit_time == "", "symbol"].unique().tolist() + [sym]
    rets: Dict[str, pd.Series] = {}
    for s in open_syms:
        d = ctx.data_fetcher.get_daily_df(ctx, s)
        if d is None or d.empty:
            continue
        # Handle DataFrame with MultiIndex columns (symbol, field) or single-level
        if isinstance(d.columns, pd.MultiIndex):
            if (s, "Close") in d.columns:
                series = d[(s, "Close")].pct_change().dropna()
            else:
                continue
        else:
            series = d["Close"].pct_change().dropna()
        if not series.empty:
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

def get_sector(symbol: str) -> str:
    if symbol in _SECTOR_CACHE:
        return _SECTOR_CACHE[symbol]
    try:
        sector = yf.Ticker(symbol).info.get("sector", "Unknown")
    except Exception:
        sector = "Unknown"
    _SECTOR_CACHE[symbol] = sector
    return sector

def sector_exposure(ctx: BotContext) -> Dict[str, float]:
    """Return current portfolio exposure by sector as fraction of equity."""
    try:
        positions = ctx.api.list_positions()
    except Exception:
        return {}
    try:
        total = float(ctx.api.get_account().portfolio_value)
    except Exception:
        total = 0.0
    exposure: Dict[str, float] = {}
    for pos in positions:
        qty = abs(int(getattr(pos, "qty", 0)))
        price = float(getattr(pos, "current_price", 0) or getattr(pos, "avg_entry_price", 0) or 0)
        sec = get_sector(getattr(pos, "symbol", ""))
        val = qty * price
        exposure[sec] = exposure.get(sec, 0.0) + val
    if total <= 0:
        return {k: 0.0 for k in exposure}
    return {k: v / total for k, v in exposure.items()}

def sector_exposure_ok(ctx: BotContext, symbol: str, qty: int, price: float) -> bool:
    """Return True if adding qty*price of symbol keeps sector exposure within cap."""
    sec = get_sector(symbol)
    exposures = sector_exposure(ctx)
    try:
        total = float(ctx.api.get_account().portfolio_value)
    except Exception:
        total = 0.0
    projected = exposures.get(sec, 0.0) + ((qty * price) / total if total > 0 else 0.0)
    return projected <= SECTOR_EXPOSURE_CAP

# ─── K. SIZING & EXECUTION HELPERS ─────────────────────────────────────────────
def is_within_entry_window(ctx: BotContext) -> bool:
    """Return True if current time is during regular Eastern trading hours."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    start = dt_time(9, 30)
    end = dt_time(16, 0)
    if not (start <= now_et.time() <= end):
        logger.info(
            "SKIP_ENTRY_WINDOW",
            extra={"start": start, "end": end, "now": now_et.time()},
        )
        return False
    if _STREAK_HALT_UNTIL and datetime.now(PACIFIC) < _STREAK_HALT_UNTIL:
        logger.info("SKIP_STREAK_HALT", extra={"until": _STREAK_HALT_UNTIL})
        return False
    return True

def scaled_atr_stop(
    entry_price: float,
    atr: float,
    now: datetime,
    market_open: datetime,
    market_close: datetime,
    max_factor: float = 2.0,
    min_factor: float = 0.5
) -> Tuple[float, float]:
    total = (market_close - market_open).total_seconds()
    elapsed = (now - market_open).total_seconds()
    α = max(0, min(1, 1 - elapsed / total))
    factor = min_factor + α * (max_factor - min_factor)
    stop = entry_price - factor * atr
    take = entry_price + factor * atr
    return stop, take

def fractional_kelly_size(
    ctx: BotContext,
    balance: float,
    price: float,
    atr: float,
    win_prob: float,
    payoff_ratio: float = 1.5
) -> int:
    # Volatility throttle: if SPY ATR > 2σ, cut kelly in half
    if is_high_vol_thr_spy():
        base_frac = ctx.kelly_fraction * 0.5
    else:
        base_frac = ctx.kelly_fraction

    if not os.path.exists(PEAK_EQUITY_FILE):
        with portalocker.Lock(PEAK_EQUITY_FILE, 'w', timeout=5) as f:
            f.write(str(balance))
        peak_equity = balance
    else:
        with portalocker.Lock(PEAK_EQUITY_FILE, 'r+', timeout=5) as f:
            content = f.read().strip()
            peak_equity = float(content) if content else balance
            if balance > peak_equity:
                f.seek(0); f.truncate(); f.write(str(balance))
                peak_equity = balance

    drawdown = (peak_equity - balance) / peak_equity
    if drawdown > 0.10:
        frac = 0.3
    elif drawdown > 0.05:
        frac = 0.45
    else:
        frac = base_frac

    edge = win_prob - (1 - win_prob) / payoff_ratio
    kelly = max(edge / payoff_ratio, 0) * frac
    dollars_to_risk = kelly * balance
    if atr <= 0:
        return 1

    raw_pos = dollars_to_risk / atr
    cap_pos = (balance * CAPITAL_CAP) / price if price > 0 else 0
    risk_cap = (balance * DOLLAR_RISK_LIMIT) / atr if atr > 0 else raw_pos
    size = int(min(raw_pos, cap_pos, risk_cap, MAX_POSITION_SIZE))
    return max(size, 1)

def vol_target_position_size(
    cash: float,
    price: float,
    returns: np.ndarray,
    target_vol: float = 0.02
) -> int:
    sigma = np.std(returns)
    if sigma <= 0 or price <= 0:
        return 1
    dollar_alloc = cash * (target_vol / sigma)
    qty = int(dollar_alloc / price)
    return max(qty, 1)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(APIError),
)
def submit_order(ctx: BotContext, symbol: str, qty: int, side: str) -> Optional[Order]:
    """Submit an order with microstructure-aware logic."""

    def _send(qty_slice: int) -> Optional[Order]:
        attempt = 0
        while attempt < 3:
            attempt += 1
            try:
                quote = ctx.api.get_latest_quote(symbol)
                bid = float(getattr(quote, "bid_price", 0) or 0)
                ask = float(getattr(quote, "ask_price", 0) or 0)
            except Exception as e:
                logger.warning(f"[submit_order] quote fetch failed (attempt {attempt}) for {symbol}: {e}")
                bid = ask = 0.0

            now_utc = pd.Timestamp.utcnow()
            regular = in_trading_hours(now_utc)

            order_kwargs: dict[str, Any] = {
                "symbol": symbol,
                "qty": qty_slice,
                "side": side,
            }
            expected_price: Optional[float] = None

            if regular:
                spread = (ask - bid) if ask and bid else 0.0
                mid = (ask + bid) / 2 if ask and bid else None
                order_kwargs["time_in_force"] = "gtc"
                if mid:
                    if side.lower() == "buy":
                        limit_price = round(min(mid + 0.01, ask or mid), 2)
                    else:
                        limit_price = round(max(mid - 0.01, bid or mid), 2)
                else:
                    limit_price = None

                if spread > 0.03 and limit_price:
                    order_kwargs.update({"type": "limit", "limit_price": limit_price})
                    expected_price = limit_price
                else:
                    order_kwargs["type"] = "market"
                    expected_price = ask if side.lower() == "buy" else bid
            else:
                px = bid if side.lower() == "buy" else ask
                if px <= 0:
                    logger.warning("OFF_HOURS_NO_PRICE", extra={"symbol": symbol})
                    order_kwargs.update({"type": "market", "time_in_force": "opg"})
                    expected_price = None
                else:
                    limit_price = round(px, 2)
                    order_kwargs.update({
                        "type": "limit",
                        "limit_price": limit_price,
                        "time_in_force": "day",
                        "extended_hours": True,
                    })
                    expected_price = limit_price

            try:
                logger.info(
                    "ORDER_SENT",
                    extra={
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": symbol,
                        "side": side,
                        "qty": qty_slice,
                        "order_type": order_kwargs.get("type", "market"),
                    },
                )
                order = ctx.api.submit_order(**order_kwargs)
                if order_kwargs.get("type") == "limit":
                    pytime.sleep(5)
                    try:
                        od_check = ctx.api.get_order(order.id)
                        status = getattr(od_check, "status", "")
                        filled_q = int(float(getattr(od_check, "filled_qty", 0) or 0))
                        if status in {"new", "accepted", "partially_filled"} and filled_q < qty_slice:
                            ctx.api.cancel_order(order.id)
                            order_kwargs.pop("limit_price", None)
                            order_kwargs["type"] = "market"
                            order = ctx.api.submit_order(**order_kwargs)
                    except Exception:
                        pass
                if not getattr(order, "id", None):
                    logger.error(
                        "ORDER_SUBMISSION_NO_ID",
                        extra={"symbol": symbol, "response": str(order)},
                    )
                fill_price = float(
                    getattr(order, "filled_avg_price", expected_price or 0) or 0
                )
                slip = ((fill_price - expected_price) * 100) if expected_price else 0.0
                slippage_total.inc(abs(slip))
                slippage_count.inc()
                _slippage_log.append((symbol, expected_price, fill_price, datetime.now(timezone.utc)))
                with slippage_lock:
                    with open(SLIPPAGE_LOG_FILE, "a", newline="") as sf:
                        csv.writer(sf).writerow([
                            datetime.now(timezone.utc).isoformat(),
                            symbol,
                            expected_price,
                            fill_price,
                            slip,
                        ])
                logger.info(
                    "ORDER_ACK",
                    extra={
                        "symbol": symbol,
                        "order_id": getattr(order, "id", ""),
                        "status": getattr(order, "status", ""),
                        "expected_price": expected_price,
                        "fill_price": fill_price,
                        "slippage_cents": slip,
                    },
                )
                filled_qty = int(float(getattr(order, "filled_qty", 0) or 0))
                if filled_qty < qty_slice:
                    logger.info(
                        "ORDER_PARTIAL_FILL",
                        extra={"symbol": symbol, "expected": qty_slice, "filled": filled_qty},
                    )
                Thread(
                    target=poll_order_fill_status,
                    args=(ctx, getattr(order, "id", "")),
                    daemon=True,
                ).start()
                orders_total.inc()
                return order
            except APIError as e:
                msg = str(e).lower()
                logger.warning(f"[submit_order] APIError (attempt {attempt}) for {symbol}: {e}")
                if "too many requests" in msg or "rate limit" in msg:
                    sleep_time = 60
                    logger.warning(f"[submit_order] rate limit hit; sleeping {sleep_time}s")
                    pytime.sleep(sleep_time)
                else:
                    pytime.sleep(attempt)
            except Exception as e:
                logger.exception(f"[submit_order] unexpected error attempt {attempt} for {symbol}: {e}")
                pytime.sleep(attempt)
        # final fallback
        try:
            logger.warning(f"FALLBACK_MARKET_ORDER for {symbol}")
            order = ctx.api.submit_order(
                symbol=symbol,
                qty=qty_slice,
                side=side,
                type="market",
                time_in_force="gtc",
            )
            logger.info(
                "ORDER_ACK",
                extra={
                    "symbol": symbol,
                    "order_id": getattr(order, "id", ""),
                    "status": getattr(order, "status", ""),
                },
            )
            Thread(
                target=poll_order_fill_status,
                args=(ctx, getattr(order, "id", "")),
                daemon=True,
            ).start()
            return order
        except Exception:
            return None

    # Handle slicing
    if qty > 50:
        remaining = qty
        last_order = None
        while remaining > 0:
            slice_qty = min(max(1, int(qty * random.uniform(0.1, 0.2))), remaining)
            last_order = _send(slice_qty)
            filled = int(float(getattr(last_order, "filled_qty", slice_qty) or slice_qty)) if last_order else 0
            remaining -= filled
            if remaining > 0:
                pytime.sleep(random.uniform(2, 5))
        return last_order
    else:
        return _send(qty)

def poll_order_fill_status(ctx: BotContext, order_id: str, timeout: int = 120) -> None:
    """Poll Alpaca for order fill status until it is no longer open."""
    start = pytime.time()
    while pytime.time() - start < timeout:
        try:
            od = ctx.api.get_order(order_id)
            status = getattr(od, "status", "")
            filled = getattr(od, "filled_qty", "0")
            if status not in {"new", "accepted", "partially_filled"}:
                logger.info(
                    "ORDER_FINAL_STATUS",
                    extra={"order_id": order_id, "status": status, "filled_qty": filled},
                )
                return
        except Exception as e:
            logger.warning(f"[poll_order_fill_status] failed for {order_id}: {e}")
            return
        pytime.sleep(3)

def send_exit_order(ctx: BotContext, symbol: str, exit_qty: int, price: float, reason: str) -> None:
    logger.info(
        f"EXIT_SIGNAL | symbol={symbol}  reason={reason}  exit_qty={exit_qty}  price={price}"
    )
    try:
        pos = ctx.api.get_position(symbol)
        held_qty = int(pos.qty)
    except Exception:
        held_qty = 0

    if held_qty < exit_qty:
        logger.warning(
            f"No shares available to exit for {symbol} (requested {exit_qty}, have {held_qty})"
        )
        return

    if price <= 0.0:
        ctx.api.submit_order(symbol=symbol, qty=exit_qty, side="sell", type="market")
        return

    limit_order = ctx.api.submit_order(
        symbol=symbol,
        qty=exit_qty,
        side="sell",
        type="limit",
        limit_price=price,
    )
    pytime.sleep(5)
    try:
        o2 = ctx.api.get_order(limit_order.id)
        if getattr(o2, "status", "") in {"new", "accepted", "partially_filled"}:
            ctx.api.cancel_order(limit_order.id)
            ctx.api.submit_order(symbol=symbol, qty=exit_qty, side="sell", type="market")
    except Exception as e:
        logger.warning(
            f"[send_exit_order] couldn\u2019t check/cancel order {getattr(limit_order, 'id', '')}: {e}"
        )

def twap_submit(
    ctx: BotContext,
    symbol: str,
    total_qty: int,
    side: str,
    window_secs: int = 600,
    n_slices: int = 10,
) -> None:
    slice_qty = total_qty // n_slices
    wait_secs = window_secs / n_slices
    for i in range(n_slices):
        try:
            submit_order(ctx, symbol, slice_qty, side)
        except Exception as e:
            logger.exception(f"[TWAP] slice {i+1}/{n_slices} failed: {e}")
            break
        pytime.sleep(wait_secs)

def vwap_pegged_submit(
    ctx: BotContext,
    symbol: str,
    total_qty: int,
    side: str,
    duration: int = 300
) -> None:
    start_time = pytime.time()
    placed = 0
    while placed < total_qty and pytime.time() - start_time < duration:
        df = ctx.data_fetcher.get_minute_df(ctx, symbol)
        if df is None or df.empty:
            logger.warning("[VWAP] missing bars, aborting VWAP slice", extra={"symbol": symbol})
            break
        vwap_price = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"]).iloc[-1]
        try:
            quote = ctx.api.get_latest_quote(symbol)
            spread = (quote.ask_price - quote.bid_price) if quote.ask_price and quote.bid_price else 0.0
        except Exception:
            spread = 0.0
        if spread > 0.05:
            slice_qty = max(1, int((total_qty - placed) * 0.5))
        else:
            slice_qty = min(max(1, total_qty // 10), total_qty - placed)
        order = None
        for attempt in range(3):
            try:
                logger.info(
                    "ORDER_SENT",
                    extra={
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": symbol,
                        "side": side,
                        "qty": slice_qty,
                        "order_type": "limit",
                    },
                )
                order = ctx.api.submit_order(
                    symbol=symbol,
                    qty=slice_qty,
                    side=side,
                    type="limit",
                    time_in_force="ioc",
                    limit_price=round(vwap_price, 2),
                )
                logger.info(
                    "ORDER_ACK",
                    extra={
                        "symbol": symbol,
                        "order_id": getattr(order, "id", ""),
                        "status": getattr(order, "status", ""),
                    },
                )
                Thread(
                    target=poll_order_fill_status,
                    args=(ctx, getattr(order, "id", "")),
                    daemon=True,
                ).start()
                fill_price = float(getattr(order, "filled_avg_price", 0) or 0)
                if fill_price > 0:
                    slip = (fill_price - vwap_price) * 100
                    slippage_total.inc(abs(slip))
                    slippage_count.inc()
                    _slippage_log.append((symbol, vwap_price, fill_price, datetime.now(timezone.utc)))
                    with slippage_lock:
                        with open(SLIPPAGE_LOG_FILE, "a", newline="") as sf:
                            csv.writer(sf).writerow([
                                datetime.now(timezone.utc).isoformat(),
                                symbol,
                                vwap_price,
                                fill_price,
                                slip,
                            ])
                orders_total.inc()
                break
            except APIError as e:
                logger.warning(f"[VWAP] APIError attempt {attempt+1} for {symbol}: {e}")
                pytime.sleep(attempt + 1)
            except Exception as e:
                logger.exception(f"[VWAP] slice attempt {attempt+1} failed: {e}")
                pytime.sleep(attempt + 1)
        if order is None:
            break
        placed += slice_qty
        pytime.sleep(duration / 10)

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

def pov_submit(
    ctx: BotContext,
    symbol: str,
    total_qty: int,
    side: str,
    cfg: SliceConfig = DEFAULT_SLICE_CFG,
) -> bool:
    placed = 0
    retries = 0
    interval = cfg.sleep_interval
    while placed < total_qty:
        df = ctx.data_fetcher.get_minute_df(ctx, symbol)
        if df is None or df.empty:
            retries += 1
            if retries > cfg.max_retries:
                logger.warning(f"[pov_submit] no minute data after {cfg.max_retries} retries, aborting", extra={"symbol": symbol})
                return False
            logger.warning(f"[pov_submit] missing bars, retry {retries}/{cfg.max_retries} in {interval:.1f}s", extra={"symbol": symbol})
            sleep_time = interval * (0.8 + 0.4 * random.random())
            pytime.sleep(sleep_time)
            interval = min(interval * cfg.backoff_factor, cfg.max_backoff_interval)
            continue
        retries = 0
        interval = cfg.sleep_interval

        try:
            quote = ctx.api.get_latest_quote(symbol)
            spread = (quote.ask_price - quote.bid_price) if quote.ask_price and quote.bid_price else 0.0
        except Exception:
            spread = 0.0

        vol = df["Volume"].iloc[-1]
        if spread > 0.05:
            slice_qty = min(int(vol * cfg.pct * 0.5), total_qty - placed)
        else:
            slice_qty = min(int(vol * cfg.pct), total_qty - placed)

        if slice_qty < 1:
            logger.debug(f"[pov_submit] slice_qty<1 (vol={vol}), waiting", extra={"symbol": symbol})
            pytime.sleep(cfg.sleep_interval * (0.8 + 0.4 * random.random()))
            continue
        try:
            od = submit_order(ctx, symbol, slice_qty, side)
        except Exception as e:
            logger.exception(f"[pov_submit] submit_order failed on slice, aborting: {e}", extra={"symbol": symbol})
            return False
        placed += slice_qty
        logger.info("POV_SLICE_PLACED", extra={"symbol": symbol, "slice_qty": slice_qty, "placed": placed})
        pytime.sleep(cfg.sleep_interval * (0.8 + 0.4 * random.random()))
    logger.info("POV_SUBMIT_COMPLETE", extra={"symbol": symbol, "placed": placed})
    return True

def maybe_pyramid(ctx: BotContext, symbol: str, entry_price: float, current_price: float, atr: float, prob: float):
    """Add to a winning position when probability remains high."""
    profit = (current_price - entry_price) if entry_price else 0
    if profit > 2 * atr and prob >= 0.75:
        try:
            pos = ctx.api.get_position(symbol)
            qty = int(abs(int(pos.qty)) * 0.5)
            if qty > 0:
                submit_order(ctx, symbol, qty, "buy")
                logger.info("PYRAMIDED", extra={"symbol": symbol, "qty": qty})
        except Exception as e:
            logger.exception(f"[maybe_pyramid] failed for {symbol}: {e}")

def update_trailing_stop(
    ctx: BotContext,
    ticker: str,
    price: float,
    qty: int,
    atr: float,
) -> str:
    factor = 1.0 if is_high_vol_regime() else TRAILING_FACTOR
    te = ctx.trailing_extremes
    if qty > 0:
        with targets_lock:
            te[ticker] = max(te.get(ticker, price), price)
        if price < te[ticker] - factor * atr:
            return "exit_long"
    elif qty < 0:
        with targets_lock:
            te[ticker] = min(te.get(ticker, price), price)
        if price > te[ticker] + factor * atr:
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
    df_daily = ctx.data_fetcher.get_daily_df(ctx, symbol)
    avg_vol = df_daily["Volume"].tail(20).mean() if df_daily is not None else 0
    cap_pct = 0.05 if avg_vol > 1e7 else 0.03
    cap_sz = int((cash * cap_pct) / price) if price > 0 else 0
    df = ctx.data_fetcher.get_daily_df(ctx, symbol)
    rets = df["Close"].pct_change().dropna().values if df is not None and not df.empty else np.array([0.0])
    kelly_sz = fractional_kelly_size(ctx, cash, price, atr, win_prob)
    vol_sz = vol_target_position_size(cash, price, rets, target_vol=0.02)
    base = int(min(kelly_sz, vol_sz, cap_sz, MAX_POSITION_SIZE))
    factor = max(0.5, min(1.5, 1 + (win_prob - 0.5)))
    size = int(base * factor)
    return max(size, 1)

def execute_entry(ctx: BotContext, symbol: str, qty: int, side: str) -> None:
    buying_pw = float(ctx.api.get_account().buying_power)
    if buying_pw <= 0:
        logger.info("NO_BUYING_POWER", extra={"symbol": symbol})
        return
    if qty <= 0:
        logger.warning("ZERO_QTY", extra={"symbol": symbol})
        return
    if POV_SLICE_PCT > 0 and qty > SLICE_THRESHOLD:
        logger.info("POV_SLICE_ENTRY", extra={"symbol": symbol, "qty": qty})
        pov_submit(ctx, symbol, qty, side)
    elif qty > SLICE_THRESHOLD:
        logger.info("VWAP_SLICE_ENTRY", extra={"symbol": symbol, "qty": qty})
        vwap_pegged_submit(ctx, symbol, qty, side)
    else:
        logger.info("MARKET_ENTRY", extra={"symbol": symbol, "qty": qty})
        submit_order(ctx, symbol, qty, side)

    raw = ctx.data_fetcher.get_minute_df(ctx, symbol)
    if raw is None or raw.empty:
        logger.warning("NO_MINUTE_BARS_POST_ENTRY", extra={"symbol": symbol})
        return
    df_ind = prepare_indicators(raw, freq="intraday")
    if df_ind.empty:
        logger.warning("INSUFFICIENT_INDICATORS_POST_ENTRY", extra={"symbol": symbol})
        return
    entry_price = df_ind["Close"].iloc[-1]
    ctx.trade_logger.log_entry(symbol, entry_price, qty, side, "", "")

    now_pac = datetime.now(PACIFIC)
    mo = datetime.combine(now_pac.date(), ctx.market_open, PACIFIC)
    mc = datetime.combine(now_pac.date(), ctx.market_close, PACIFIC)
    if is_high_vol_regime():
        tp_factor = TAKE_PROFIT_FACTOR * 1.1
    else:
        tp_factor = TAKE_PROFIT_FACTOR
    stop, take = scaled_atr_stop(
        entry_price,
        df_ind["atr"].iloc[-1],
        now_pac, mo, mc,
        max_factor=tp_factor,
        min_factor=0.5
    )
    with targets_lock:
        ctx.stop_targets[symbol] = stop
        ctx.take_profit_targets[symbol] = take

def execute_exit(ctx: BotContext, symbol: str, qty: int) -> None:
    if qty <= 0:
        return
    raw = ctx.data_fetcher.get_minute_df(ctx, symbol)
    exit_price = raw["Close"].iloc[-1] if raw is not None and not raw.empty else 0.0
    send_exit_order(ctx, symbol, qty, exit_price, "manual_exit")
    ctx.trade_logger.log_exit(symbol, exit_price)
    on_trade_exit_rebalance(ctx)
    with targets_lock:
        ctx.take_profit_targets.pop(symbol, None)
        ctx.stop_targets.pop(symbol, None)

def exit_all_positions(ctx: BotContext) -> None:
    for pos in ctx.api.list_positions():
        qty = abs(int(pos.qty))
        if qty:
            send_exit_order(ctx, pos.symbol, qty, 0.0, "eod_exit")
            logger.info("EOD_EXIT", extra={"symbol": pos.symbol, "qty": qty})

# ─── L. SIGNAL & TRADE LOGIC ───────────────────────────────────────────────────
def signal_and_confirm(ctx: BotContext, symbol: str, df: pd.DataFrame, model) -> Tuple[int, float, str]:
    sig, conf, strat = ctx.signal_manager.evaluate(ctx, df, symbol, model)
    if sig == -1 or conf < CONF_THRESHOLD:
        logger.debug("SKIP_LOW_SIGNAL", extra={"symbol": symbol, "sig": sig, "conf": conf})
        return -1, 0.0, ""
    return sig, conf, strat

def pre_trade_checks(
    ctx: BotContext,
    symbol: str,
    balance: float,
    regime_ok: bool
) -> bool:
    # Streak kill-switch check
    if _STREAK_HALT_UNTIL and datetime.now(PACIFIC) < _STREAK_HALT_UNTIL:
        logger.info("SKIP_STREAK_HALT", extra={"symbol": symbol, "until": _STREAK_HALT_UNTIL})
        return False
    if check_pdt_rule(ctx):
        logger.info("SKIP_PDT_RULE", extra={"symbol": symbol})
        return False
    if check_halt_flag():
        logger.info("SKIP_HALT_FLAG", extra={"symbol": symbol})
        return False
    if check_daily_loss(ctx):
        logger.info("SKIP_DAILY_LOSS", extra={"symbol": symbol})
        return False
    if check_weekly_loss(ctx):
        logger.info("SKIP_WEEKLY_LOSS", extra={"symbol": symbol})
        return False
    if too_many_positions(ctx):
        logger.info("SKIP_TOO_MANY_POSITIONS", extra={"symbol": symbol})
        return False
    if too_correlated(symbol):
        logger.info("SKIP_HIGH_CORRELATION", extra={"symbol": symbol})
        return False
    return ctx.data_fetcher.get_daily_df(ctx, symbol) is not None

def should_enter(
    ctx: BotContext,
    symbol: str,
    balance: float,
    regime_ok: bool
) -> bool:
    return pre_trade_checks(ctx, symbol, balance, regime_ok) and is_within_entry_window(ctx)

def should_exit(ctx: BotContext, symbol: str, price: float, atr: float) -> Tuple[bool, int, str]:
    try:
        pos = ctx.api.get_position(symbol)
        current_qty = int(pos.qty)
    except Exception:
        current_qty = 0

    stop = ctx.stop_targets.get(symbol)
    if stop is not None:
        if current_qty > 0 and price <= stop:
            return True, abs(current_qty), "stop_loss"
        if current_qty < 0 and price >= stop:
            return True, abs(current_qty), "stop_loss"

    tp = ctx.take_profit_targets.get(symbol)
    if current_qty > 0 and tp and price >= tp:
        exit_qty = max(int(abs(current_qty) * SCALING_FACTOR), 1)
        return True, exit_qty, "take_profit"
    if current_qty < 0 and tp and price <= tp:
        exit_qty = max(int(abs(current_qty) * SCALING_FACTOR), 1)
        return True, exit_qty, "take_profit"

    action = update_trailing_stop(ctx, symbol, price, current_qty, atr)
    if (action == "exit_long" and current_qty > 0) or (action == "exit_short" and current_qty < 0):
        return True, abs(current_qty), "trailing_stop"

    return False, 0, ""

def _safe_trade(
    ctx: BotContext,
    symbol: str,
    balance: float,
    model: RandomForestClassifier,
    regime_ok: bool
) -> None:
    try:
        trade_logic(ctx, symbol, balance, model, regime_ok)
    except RetryError as e:
        logger.warning(f"[trade_logic] retries exhausted for {symbol}: {e}", extra={"symbol": symbol})
    except APIError as e:
        msg = str(e).lower()
        if "insufficient buying power" in msg or "potential wash trade" in msg:
            logger.warning(f"[trade_logic] skipping {symbol} due to APIError: {e}", extra={"symbol": symbol})
        else:
            logger.exception(f"[trade_logic] APIError for {symbol}: {e}")
    except Exception:
        logger.exception(f"[trade_logic] unhandled exception for {symbol}")

def trade_logic(
    ctx: BotContext,
    symbol: str,
    balance: float,
    model,
    regime_ok: bool
) -> None:
    """
    Core per-symbol logic: fetch data, compute features, evaluate signals, enter/exit orders.
    """
    logger.info(f"PROCESSING_SYMBOL | symbol={symbol}")

    # Run pre-trade checks that enforce PDT, halt flags, market regime, etc.
    if not should_enter(ctx, symbol, balance, regime_ok):
        logger.debug("SKIP_PRE_TRADE_CHECKS", extra={"symbol": symbol})
        return

    raw_df = ctx.data_fetcher.get_minute_df(ctx, symbol)
    if raw_df is None or raw_df.empty:
        logger.info(f"SKIP_NO_RAW_DATA | symbol={symbol}")
        return

    feat_df = prepare_indicators(raw_df, freq="intraday")
    if feat_df.empty:
        logger.debug(f"SKIP_INSUFFICIENT_FEATURES | symbol={symbol}")
        return

    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    else:
        feature_names = ["rsi", "macd", "atr", "vwap", "macds",
                         "ichimoku_conv", "ichimoku_base", "stochrsi"]

    missing = [f for f in feature_names if f not in feat_df.columns]
    if missing:
        logger.info(f"SKIP_MISSING_FEATURES | symbol={symbol}  missing={missing}")
        return

    sig, conf, strat = ctx.signal_manager.evaluate(ctx, feat_df, symbol, model)
    comp_list = [
        {"signal": lab, "flag": s, "weight": w}
        for s, w, lab in ctx.signal_manager.last_components
    ]
    logger.debug(f"COMPONENTS | symbol={symbol}  components={comp_list!r}")

    final_score = sum(s * w for s, w, _ in ctx.signal_manager.last_components)
    # ←← Log actual numeric values in the message itself:
    logger.info(
        f"SIGNAL_RESULT | symbol={symbol}  final_score={final_score:.4f}  confidence={conf:.4f}"
    )

    try:
        pos = ctx.api.get_position(symbol)
        current_qty = int(pos.qty)
    except Exception:
        current_qty = 0

    # Exit: bearish reversal for longs
    if final_score < 0 and current_qty > 0 and abs(conf) >= CONF_THRESHOLD:
        price = feat_df["Close"].iloc[-1]
        logger.info(
            f"SIGNAL_REVERSAL_EXIT | symbol={symbol}  final_score={final_score:.4f}  confidence={conf:.4f}"
        )
        send_exit_order(ctx, symbol, current_qty, price, "reversal")
        ctx.trade_logger.log_exit(symbol, price)
        with targets_lock:
            ctx.stop_targets.pop(symbol, None)
            ctx.take_profit_targets.pop(symbol, None)
        return

    # Exit: bullish reversal for shorts
    if final_score > 0 and current_qty < 0 and abs(conf) >= CONF_THRESHOLD:
        price = feat_df["Close"].iloc[-1]
        logger.info(
            f"SIGNAL_BULLISH_EXIT | symbol={symbol}  final_score={final_score:.4f}  confidence={conf:.4f}"
        )
        send_exit_order(ctx, symbol, abs(current_qty), price, "reversal")
        ctx.trade_logger.log_exit(symbol, price)
        with targets_lock:
            ctx.stop_targets.pop(symbol, None)
            ctx.take_profit_targets.pop(symbol, None)
        return

    # Entry: bullish
    if final_score > 0 and conf >= BUY_THRESHOLD and current_qty == 0:
        current_price = feat_df["Close"].iloc[-1]
        target_weight = ctx.portfolio_weights.get(symbol, 0.0)
        raw_qty = int(balance * target_weight / current_price) if current_price > 0 else 0

        if raw_qty <= 0:
            logger.debug(f"SKIP_NO_QTY | symbol={symbol}")
            return

        logger.info(
            f"SIGNAL_BUY | symbol={symbol}  final_score={final_score:.4f}  "
            f"confidence={conf:.4f}  qty={raw_qty}"
        )

        if not sector_exposure_ok(ctx, symbol, raw_qty, current_price):
            logger.info("SKIP_SECTOR_CAP", extra={"symbol": symbol})
            return

        order = submit_order(ctx, symbol, raw_qty, "buy")
        if order is None:
            logger.debug(f"TRADE_LOGIC_NO_ORDER | symbol={symbol}")
        else:
            logger.debug(f"TRADE_LOGIC_ORDER_PLACED | symbol={symbol}  order_id={order.id}")
            ctx.trade_logger.log_entry(symbol, current_price, raw_qty, "buy", strat)

            now_pac = datetime.now(PACIFIC)
            mo = datetime.combine(now_pac.date(), ctx.market_open, PACIFIC)
            mc = datetime.combine(now_pac.date(), ctx.market_close, PACIFIC)

            if is_high_vol_regime():
                tp_factor = TAKE_PROFIT_FACTOR * 1.1
            else:
                tp_factor = TAKE_PROFIT_FACTOR

            stop, take = scaled_atr_stop(
                entry_price=current_price,
                atr=feat_df["atr"].iloc[-1],
                now=now_pac, market_open=mo, market_close=mc,
                max_factor=tp_factor, min_factor=0.5
            )
            with targets_lock:
                ctx.stop_targets[symbol] = stop
                ctx.take_profit_targets[symbol] = take

        return

    # Entry: bearish short
    if final_score < 0 and conf >= BUY_THRESHOLD and current_qty == 0:
        current_price = feat_df["Close"].iloc[-1]
        atr = feat_df["atr"].iloc[-1]
        qty = calculate_entry_size(ctx, symbol, current_price, atr, conf)

        try:
            asset = ctx.api.get_asset(symbol)
            if hasattr(asset, "shortable") and not asset.shortable:
                logger.info(f"SKIP_NOT_SHORTABLE | symbol={symbol}")
                return
            avail = getattr(asset, "shortable_shares", None)
            if avail is not None:
                qty = min(qty, int(avail))
        except Exception:
            pass

        if qty <= 0:
            logger.debug(f"SKIP_NO_QTY | symbol={symbol}")
            return

        logger.info(
            f"SIGNAL_SHORT | symbol={symbol}  final_score={final_score:.4f}  "
            f"confidence={conf:.4f}  qty={qty}"
        )
        if not sector_exposure_ok(ctx, symbol, qty, current_price):
            logger.info("SKIP_SECTOR_CAP", extra={"symbol": symbol})
            return

        order = submit_order(ctx, symbol, qty, "sell")
        if order is None:
            logger.debug(f"TRADE_LOGIC_NO_ORDER | symbol={symbol}")
        else:
            logger.debug(f"TRADE_LOGIC_ORDER_PLACED | symbol={symbol}  order_id={order.id}")
            ctx.trade_logger.log_entry(symbol, current_price, qty, "sell", strat)

            now_pac = datetime.now(PACIFIC)
            mo = datetime.combine(now_pac.date(), ctx.market_open, PACIFIC)
            mc = datetime.combine(now_pac.date(), ctx.market_close, PACIFIC)

            if is_high_vol_regime():
                tp_factor = TAKE_PROFIT_FACTOR * 1.1
            else:
                tp_factor = TAKE_PROFIT_FACTOR

            long_stop, long_take = scaled_atr_stop(
                entry_price=current_price,
                atr=atr,
                now=now_pac, market_open=mo, market_close=mc,
                max_factor=tp_factor, min_factor=0.5
            )
            stop, take = long_take, long_stop
            with targets_lock:
                ctx.stop_targets[symbol] = stop
                ctx.take_profit_targets[symbol] = take

        return

    # If holding, check for stops/take/trailing
    if current_qty != 0:
        price = feat_df["Close"].iloc[-1]
        atr   = feat_df["atr"].iloc[-1]
        should_exit_flag, exit_qty, reason = should_exit(ctx, symbol, price, atr)
        if should_exit_flag and exit_qty > 0:
            logger.info(
                f"EXIT_SIGNAL | symbol={symbol}  reason={reason}  "
                f"exit_qty={exit_qty}  price={price:.4f}"
            )
            send_exit_order(ctx, symbol, exit_qty, price, reason)
            ctx.trade_logger.log_exit(symbol, price)
            try:
                pos_after = ctx.api.get_position(symbol)
                if int(pos_after.qty) == 0:
                    with targets_lock:
                        ctx.stop_targets.pop(symbol, None)
                        ctx.take_profit_targets.pop(symbol, None)
            except Exception:
                pass
        else:
            try:
                pos = ctx.api.get_position(symbol)
                entry_price = float(pos.avg_entry_price)
                maybe_pyramid(ctx, symbol, entry_price, price, atr, conf)
            except Exception:
                pass
        return

    # Else hold / no action
    logger.info(
        f"SKIP_LOW_OR_NO_SIGNAL | symbol={symbol}  "
        f"final_score={final_score:.4f}  confidence={conf:.4f}"
    )
    return

def compute_portfolio_weights(symbols: List[str]) -> Dict[str, float]:
    """
    Equal-weight portfolio, but can be overridden by PCA-based factor exposure model.
    """
    n = len(symbols)
    if n == 0:
        logger.warning("No tickers to weight—skipping.")
        return {}
    equal_weight = 1.0 / n
    weights = {s: equal_weight for s in symbols}
    logger.info("PORTFOLIO_WEIGHTS", extra={"weights": weights})
    return weights

def on_trade_exit_rebalance(ctx: BotContext) -> None:
    current = compute_portfolio_weights(list(ctx.portfolio_weights.keys()))
    old = ctx.portfolio_weights
    drift = max(abs(current[s] - old.get(s, 0)) for s in current) if current else 0
    if drift <= 0.1:
        return
    ctx.portfolio_weights = current
    total_value = float(ctx.api.get_account().portfolio_value)
    for sym, w in current.items():
        target_dollar = w * total_value
        raw = ctx.data_fetcher.get_minute_df(ctx, sym)
        price = raw["Close"].iloc[-1] if raw is not None and not raw.empty else 0.0
        if price <= 0:
            continue
        target_shares = int(target_dollar / price)
        try:
            submit_order(
                ctx,
                sym,
                abs(target_shares),
                "buy" if target_shares > 0 else "sell",
            )
        except Exception:
            logger.exception(f"Rebalance failed for {sym}")
    logger.info("PORTFOLIO_REBALANCED")

def pair_trade_signal(sym1: str, sym2: str) -> Tuple[str, int]:
    from statsmodels.tsa.stattools import coint
    df1 = ctx.data_fetcher.get_daily_df(ctx, sym1)
    df2 = ctx.data_fetcher.get_daily_df(ctx, sym2)
    if not hasattr(df1, "loc") or "Close" not in df1.columns:
        raise ValueError(f"pair_trade_signal: df1 for {sym1} is invalid or missing 'Close'")
    if not hasattr(df2, "loc") or "Close" not in df2.columns:
        raise ValueError(f"pair_trade_signal: df2 for {sym2} is invalid or missing 'Close'")
    df  = pd.concat([df1["Close"], df2["Close"]], axis=1).dropna()
    t_stat, p_value, _ = coint(df.iloc[:, 0], df.iloc[:, 1])
    if p_value < 0.05:
        beta = np.polyfit(df.iloc[:, 1], df.iloc[:, 0], 1)[0]
        spread = df.iloc[:, 0] - beta * df.iloc[:, 1]
        z = (spread - spread.mean()) / spread.std()
        z0 = z.iloc[-1]
        if z0 > 2:
            return ("short_spread", 1)
        elif z0 < -2:
            return ("long_spread", 1)
    return ("no_signal", 0)

# ─── M. UTILITIES ─────────────────────────────────────────────────────────────
def fetch_data(ctx: BotContext, symbols: List[str], period: str, interval: str) -> Optional[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    now = datetime.utcnow()
    if period.endswith("d"):
        delta = timedelta(days=int(period[:-1]))
    elif period.endswith("mo"):
        delta = timedelta(days=30 * int(period[:-2]))
    elif period.endswith("y"):
        delta = timedelta(days=365 * int(period[:-1]))
    else:
        delta = timedelta(days=7)
    unix_to = int(now.timestamp())
    unix_from = int((now - delta).timestamp())

    for batch in chunked(symbols, 3):
        for sym in batch:
            try:
                ohlc = finnhub_client.stock_candle(sym, resolution=interval, _from=unix_from, to=unix_to)
            except FinnhubAPIException as e:
                logger.warning(f"[fetch_data] {sym} error: {e}")
                continue

            if not ohlc or ohlc.get("s") != "ok":
                continue

            df_sym = pd.DataFrame({
                "Open": ohlc.get("o", []),
                "High": ohlc.get("h", []),
                "Low": ohlc.get("l", []),
                "Close": ohlc.get("c", []),
                "Volume": ohlc.get("v", []),
            }, index=pd.to_datetime(ohlc.get("t", []), unit="s"))

            df_sym.columns = pd.MultiIndex.from_product([[sym], df_sym.columns])
            frames.append(df_sym)

        pytime.sleep(random.uniform(2, 5))

    if not frames:
        return None

    return pd.concat(frames, axis=1)

class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        probs = [m.predict_proba(X) for m in self.models]
        return np.mean(probs, axis=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


def load_model(path: str = MODEL_PATH):
    rf_exists = os.path.exists(MODEL_RF_PATH)
    xgb_exists = os.path.exists(MODEL_XGB_PATH)
    lgb_exists = os.path.exists(MODEL_LGB_PATH)
    if rf_exists and xgb_exists and lgb_exists:
        models = []
        for p in [MODEL_RF_PATH, MODEL_XGB_PATH, MODEL_LGB_PATH]:
            try:
                models.append(joblib.load(p))
            except Exception as e:
                logger.exception(f"Failed to load model at {p}: {e}")
                return None
        logger.info(
            "MODEL_LOADED",
            extra={"path": f"{MODEL_RF_PATH}, {MODEL_XGB_PATH}, {MODEL_LGB_PATH}"},
        )
        return EnsembleModel(models)

    if not os.path.exists(path):
        logger.warning("MODEL_MISSING", extra={"path": path})
        return None
    try:
        model = joblib.load(path)
    except Exception as e:
        logger.exception(f"Failed to load model at {path}: {e}")
        return None
    logger.info("MODEL_LOADED", extra={"path": path})
    return model

def update_signal_weights() -> None:
    if not os.path.exists(TRADE_LOG_FILE):
        logger.warning("No trades log found; skipping weight update.")
        return
    df = pd.read_csv(TRADE_LOG_FILE).dropna(subset=["entry_price","exit_price","signal_tags"])
    df["pnl"] = (df["exit_price"] - df["entry_price"]) * df["side"].apply(lambda s: 1 if s == "buy" else -1)
    recent_cut = pd.to_datetime(df["exit_time"], errors="coerce")
    recent_mask = recent_cut >= (datetime.now(timezone.utc) - timedelta(days=30))
    df_recent = df[recent_mask]

    stats_all: Dict[str, List[float]] = {}
    stats_recent: Dict[str, List[float]] = {}
    for _, row in df.iterrows():
        for tag in row["signal_tags"].split("+"):
            stats_all.setdefault(tag, []).append(row["pnl"])
    for _, row in df_recent.iterrows():
        for tag in row["signal_tags"].split("+"):
            stats_recent.setdefault(tag, []).append(row["pnl"])

    new_weights = {}
    for tag, pnls in stats_all.items():
        overall_wr = np.mean([1 if p > 0 else 0 for p in pnls]) if pnls else 0.0
        recent_wr = np.mean([1 if p > 0 else 0 for p in stats_recent.get(tag, [])]) if stats_recent.get(tag) else overall_wr
        weight = 0.7 * recent_wr + 0.3 * overall_wr
        if recent_wr < 0.4:
            weight *= 0.5
        new_weights[tag] = round(weight, 3)

    ALPHA = 0.2
    if os.path.exists(SIGNAL_WEIGHTS_FILE):
        old = pd.read_csv(SIGNAL_WEIGHTS_FILE).set_index("signal")["weight"].to_dict()
    else:
        old = {}
    merged = {tag: round(ALPHA * w + (1 - ALPHA) * old.get(tag, w), 3) for tag, w in new_weights.items()}
    out_df = pd.DataFrame.from_dict(merged, orient="index", columns=["weight"]).reset_index()
    out_df.columns = ["signal", "weight"]
    out_df.to_csv(SIGNAL_WEIGHTS_FILE, index=False)
    logger.info("SIGNAL_WEIGHTS_UPDATED", extra={"count": len(merged)})

def run_meta_learning_weight_optimizer(
    trade_log_path: str = TRADE_LOG_FILE,
    output_path: str = SIGNAL_WEIGHTS_FILE,
    alpha: float = 1.0
):
    if not meta_lock.acquire(blocking=False):
        logger.warning("METALEARN_SKIPPED_LOCKED")
        return
    try:
        if not os.path.exists(trade_log_path):
            logger.warning("METALEARN_NO_TRADES")
            return

        df = pd.read_csv(trade_log_path).dropna(subset=["entry_price", "exit_price", "signal_tags"])
        if df.empty:
            logger.warning("METALEARN_NO_VALID_ROWS")
            return

        df["pnl"] = (df["exit_price"] - df["entry_price"]) * df["side"].map({"buy": 1, "sell": -1})
        df["outcome"] = (df["pnl"] > 0).astype(int)

        tags = sorted(set(tag for row in df["signal_tags"] for tag in row.split("+")))
        X = np.array([[int(tag in row.split("+")) for tag in tags] for row in df["signal_tags"]])
        y = df["outcome"].values

        if len(y) < len(tags):
            logger.warning("METALEARN_TOO_FEW_SAMPLES")
            return

        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X, y)
        joblib.dump(model, META_MODEL_PATH)
        logger.info("META_MODEL_TRAINED", extra={"samples": len(y)})

        weights = {tag: round(max(0, min(1, w)), 3) for tag, w in zip(tags, model.coef_)}
        out_df = pd.DataFrame(list(weights.items()), columns=["signal", "weight"])
        out_df.to_csv(output_path, index=False)
        logger.info("META_WEIGHTS_UPDATED", extra={"weights": weights})
    finally:
        meta_lock.release()

def run_bayesian_meta_learning_optimizer(
    trade_log_path: str = TRADE_LOG_FILE,
    output_path: str = SIGNAL_WEIGHTS_FILE
):
    if not meta_lock.acquire(blocking=False):
        logger.warning("METALEARN_SKIPPED_LOCKED")
        return
    try:
        if not os.path.exists(trade_log_path):
            logger.warning("METALEARN_NO_TRADES")
            return

        df = pd.read_csv(trade_log_path).dropna(subset=["entry_price", "exit_price", "signal_tags"])
        if df.empty:
            logger.warning("METALEARN_NO_VALID_ROWS")
            return

        df["pnl"] = (df["exit_price"] - df["entry_price"]) * df["side"].map({"buy": 1, "sell": -1})
        df["outcome"] = (df["pnl"] > 0).astype(int)

        tags = sorted(set(tag for row in df["signal_tags"] for tag in row.split("+")))
        X = np.array([[int(tag in row.split("+")) for tag in tags] for row in df["signal_tags"]])
        y = df["outcome"].values

        if len(y) < len(tags):
            logger.warning("METALEARN_TOO_FEW_SAMPLES")
            return

        model = BayesianRidge(fit_intercept=True, normalize=True)
        model.fit(X, y)
        joblib.dump(model, abspath("meta_model_bayes.pkl"))
        logger.info("META_MODEL_BAYESIAN_TRAINED", extra={"samples": len(y)})

        weights = {tag: round(max(0, min(1, w)), 3) for tag, w in zip(tags, model.coef_)}
        out_df = pd.DataFrame(list(weights.items()), columns=["signal", "weight"])
        out_df.to_csv(output_path, index=False)
        logger.info("META_WEIGHTS_UPDATED", extra={"weights": weights})
    finally:
        meta_lock.release()

def load_global_signal_performance(min_trades: int = 10, threshold: float = 0.4) -> Optional[Dict[str, float]]:
    if not os.path.exists(TRADE_LOG_FILE):
        logger.info("METALEARN_NO_HISTORY")
        return None
    df = pd.read_csv(TRADE_LOG_FILE).dropna(subset=["exit_price","entry_price","signal_tags"])
    df["pnl"] = (df.exit_price - df.entry_price) * df.side.apply(lambda s: 1 if s == "buy" else -1)
    results: Dict[str, List[float]] = {}
    for _, row in df.iterrows():
        for tag in row.signal_tags.split("+"):
            results.setdefault(tag, []).append(row.pnl)
    win_rates = {
        tag: round(np.mean([1 if p > 0 else 0 for p in pnls]), 3)
        for tag, pnls in results.items() if len(pnls) >= min_trades
    }
    filtered = {tag: wr for tag, wr in win_rates.items() if wr >= threshold}
    logger.info("METALEARN_FILTERED_SIGNALS", extra={"signals": list(filtered.keys()) or []})
    return filtered

def prepare_indicators(df: pd.DataFrame, freq: str = "daily") -> pd.DataFrame:
    df = df.copy()
    if df.index.name:
        df = df.reset_index().rename(columns={df.index.name: "Date"})
    else:
        df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    df["vwap"] = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"])
    df["rsi"]  = ta.rsi(df["Close"], length=14)
    df["atr"]  = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # ── New advanced indicators ───────────────────────────────────────────
    try:
        kc = ta.kc(df["High"], df["Low"], df["Close"], length=20)
        df["kc_lower"] = kc.iloc[:, 0]
        df["kc_mid"]   = kc.iloc[:, 1]
        df["kc_upper"] = kc.iloc[:, 2]
    except Exception:
        df["kc_lower"] = np.nan
        df["kc_mid"]   = np.nan
        df["kc_upper"] = np.nan

    df["atr_band_upper"] = df["Close"] + 1.5 * df["atr"]
    df["atr_band_lower"] = df["Close"] - 1.5 * df["atr"]
    df["avg_vol_20"]      = df["Volume"].rolling(20).mean()
    df["dow"]             = df.index.dayofweek

    try:
        macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        if macd is None or not isinstance(macd, pd.DataFrame):
            raise ValueError("MACD returned None")
        df["macd"]  = macd["MACD_12_26_9"]
        df["macds"] = macd["MACDs_12_26_9"]
    except Exception:
        df["macd"] = np.nan
        df["macds"] = np.nan

    # Additional indicators for richer ML features
    try:
        bb = ta.bbands(df["Close"], length=20)
        df["bb_upper"]   = bb["BBU_20_2.0"]
        df["bb_lower"]   = bb["BBL_20_2.0"]
        df["bb_percent"] = bb["BBP_20_2.0"]
    except Exception:
        df["bb_upper"] = np.nan
        df["bb_lower"] = np.nan
        df["bb_percent"] = np.nan

    try:
        adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
        df["adx"] = adx["ADX_14"]
        df["dmp"] = adx["DMP_14"]
        df["dmn"] = adx["DMN_14"]
    except Exception:
        df["adx"] = np.nan
        df["dmp"] = np.nan
        df["dmn"] = np.nan

    try:
        df["cci"] = ta.cci(df["High"], df["Low"], df["Close"], length=20)
    except Exception:
        df["cci"] = np.nan

    # Ensure numeric dtype before computing MFI
    df[["High", "Low", "Close", "Volume"]] = df[["High", "Low", "Close", "Volume"]].astype(float)
    try:
        df["mfi"] = ta.mfi(df["High"], df["Low"], df["Close"], df["Volume"], length=14)
    except Exception as e:
        logger.warning(f"prepare_indicators: failed to compute MFI: {e}")
        df["mfi"] = None

    try:
        df["tema"] = ta.tema(df["Close"], length=10)
    except Exception:
        df["tema"] = np.nan

    try:
        df["willr"] = ta.willr(df["High"], df["Low"], df["Close"], length=14)
    except Exception:
        df["willr"] = np.nan

    try:
        psar = ta.psar(df["High"], df["Low"], df["Close"])
        df["psar_long"]  = psar["PSARl_0.02_0.2"]
        df["psar_short"] = psar["PSARs_0.02_0.2"]
    except Exception:
        df["psar_long"] = np.nan
        df["psar_short"] = np.nan

    try:
        ich = ta.ichimoku(high=df["High"], low=df["Low"], close=df["Close"])
        conv = ich[0] if isinstance(ich, tuple) else ich.iloc[:, 0]
        base = ich[1] if isinstance(ich, tuple) else ich.iloc[:, 1]
        df["ichimoku_conv"] = conv.iloc[:, 0] if hasattr(conv, "iloc") else conv
        df["ichimoku_base"] = base.iloc[:, 0] if hasattr(base, "iloc") else base
    except Exception:
        df["ichimoku_conv"] = np.nan
        df["ichimoku_base"] = np.nan

    try:
        st = ta.stochrsi(df["Close"])
        df["stochrsi"] = st["STOCHRSIk_14_14_3_3"]
    except Exception:
        df["stochrsi"] = np.nan

    # --- Multi-timeframe fusion ---
    try:
        df["ret_5m"] = df["Close"].pct_change(5)
        df["ret_1h"] = df["Close"].pct_change(60)
        df["ret_d"] = df["Close"].pct_change(390)
        df["ret_w"] = df["Close"].pct_change(1950)
        df["vol_norm"] = df["Volume"].rolling(60).mean() / df["Volume"].rolling(5).mean()
        df["5m_vs_1h"] = df["ret_5m"] - df["ret_1h"]
        df["vol_5m"]  = df["Close"].pct_change().rolling(5).std()
        df["vol_1h"]  = df["Close"].pct_change().rolling(60).std()
        df["vol_d"]   = df["Close"].pct_change().rolling(390).std()
        df["vol_w"]   = df["Close"].pct_change().rolling(1950).std()
        df["vol_ratio"] = df["vol_5m"] / df["vol_1h"]
        df["mom_agg"] = df["ret_5m"] + df["ret_1h"] + df["ret_d"]
        df["lag_close_1"] = df["Close"].shift(1)
        df["lag_close_3"] = df["Close"].shift(3)
    except Exception:
        df["ret_5m"] = df["ret_1h"] = df["ret_d"] = df["ret_w"] = np.nan
        df["vol_norm"] = df["5m_vs_1h"] = np.nan
        df["vol_5m"] = df["vol_1h"] = df["vol_d"] = df["vol_w"] = np.nan
        df["vol_ratio"] = df["mom_agg"] = df["lag_close_1"] = df["lag_close_3"] = np.nan

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    required = ["vwap", "rsi", "atr", "macd", "macds", "ichimoku_conv", "ichimoku_base", "stochrsi"]
    if freq == "daily":
        df["sma_50"]  = ta.sma(df["Close"], length=50)
        df["sma_200"] = ta.sma(df["Close"], length=200)
        required += ["sma_50", "sma_200"]
        df.dropna(subset=required, how="any", inplace=True)
    else:  # intraday
        df.dropna(subset=required, how="all", inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df

def _compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat["atr"]  = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    feat["rsi"]  = ta.rsi(df["Close"], length=14)
    feat["macd"] = ta.macd(df["Close"], fast=12, slow=26, signal=9)["MACD_12_26_9"]
    feat["vol"]  = df["Close"].pct_change().rolling(14).std()
    return feat.dropna()


def detect_regime(df: pd.DataFrame) -> str:
    """Simple SMA-based market regime detection."""
    if df is None or df.empty or "Close" not in df:
        return "chop"
    close = df["Close"].astype(float)
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    if sma50.iloc[-1] > sma200.iloc[-1]:
        return "bull"
    if sma50.iloc[-1] < sma200.iloc[-1]:
        return "bear"
    return "chop"

# Train or load regime model
if os.path.exists(REGIME_MODEL_PATH):
    regime_model = pickle.load(open(REGIME_MODEL_PATH, "rb"))
else:
    today_date = date.today()
    start_date = (today_date - timedelta(days=365)).isoformat()
    bars = ctx.api.get_bars(
        REGIME_SYMBOLS[0], TimeFrame.Day,
        start=start_date, end=today_date.isoformat(),
        limit=1000, feed="iex"
    ).df
    bars.index = pd.to_datetime(bars.index).tz_localize(None)
    bars = bars.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"
    })
    feats = _compute_regime_features(bars)
    labels = (bars["Close"] > bars["Close"].rolling(200).mean()).loc[feats.index].astype(int).rename("label")
    training = feats.join(labels, how="inner").dropna()
    if len(training) >= 50:
        X = training[["atr", "rsi", "macd", "vol"]]
        y = training["label"]
        regime_model = RandomForestClassifier(n_estimators=RF_ESTIMATORS, max_depth=RF_MAX_DEPTH)
        regime_model.fit(X, y)
        pickle.dump(regime_model, open(REGIME_MODEL_PATH, "wb"))
        logger.info("REGIME_MODEL_TRAINED", extra={"rows": len(training)})
    else:
        logger.error(f"Not enough valid rows ({len(training)}) to train regime model; using dummy fallback")
        regime_model = RandomForestClassifier(n_estimators=RF_ESTIMATORS, max_depth=RF_MAX_DEPTH)

def check_market_regime() -> bool:
    # Always return True; never skip any symbol due to regime filtering.
    return True

def screen_universe(
    candidates: Sequence[str],
    ctx: BotContext,
    lookback: str = "1mo",
    interval: str = "1d",
    top_n: int = 20
) -> list[str]:
    atrs: Dict[str, float] = {}
    for sym in candidates:
        df = ctx.data_fetcher.get_daily_df(ctx, sym)
        if df is None or len(df) < ATR_LENGTH:
            continue
        series = ta.atr(df["High"], df["Low"], df["Close"], length=ATR_LENGTH)
        atr_val = series.iloc[-1] if not series.empty else np.nan
        if not pd.isna(atr_val):
            atrs[sym] = float(atr_val)
    ranked = sorted(atrs.items(), key=lambda kv: kv[1], reverse=True)
    return [sym for sym, _ in ranked[:top_n]]

def load_tickers(path: str = TICKERS_FILE) -> list[str]:
    tickers: List[str] = []
    try:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                t = row[0].strip().upper()
                if t and t not in tickers:
                    tickers.append(t)
    except Exception as e:
        logger.exception(f"[load_tickers] Failed to read {path}: {e}")
    return tickers

def daily_summary() -> None:
    if not os.path.exists(TRADE_LOG_FILE):
        logger.info("DAILY_SUMMARY_NO_TRADES")
        return
    df = pd.read_csv(TRADE_LOG_FILE).dropna(subset=["entry_price", "exit_price"])
    df["pnl"] = (df.exit_price - df.entry_price) * df["side"].map({"buy": 1, "sell": -1})
    total_trades = len(df)
    win_rate = (df.pnl > 0).mean() if total_trades else 0
    total_pnl = df.pnl.sum()
    max_dd = (df.pnl.cumsum().cummax() - df.pnl.cumsum()).max()
    logger.info(
        "DAILY_SUMMARY",
        extra={
            "trades": total_trades,
            "win_rate": f"{win_rate:.2%}",
            "pnl": total_pnl,
            "max_drawdown": max_dd
        }
    )

# ─── PCA-BASED PORTFOLIO ADJUSTMENT ─────────────────────────────────────────────
def run_daily_pca_adjustment(ctx: BotContext) -> None:
    """
    Once per day, run PCA on last 90-day returns of current universe.
    If top PC explains >40% variance and portfolio loads heavily,
    reduce those weights by 20%.
    """
    universe = list(ctx.portfolio_weights.keys())
    if not universe:
        return
    returns_df = pd.DataFrame()
    for sym in universe:
        df = ctx.data_fetcher.get_daily_df(ctx, sym)
        if df is None or len(df) < 90:
            continue
        rts = df["Close"].pct_change().tail(90).reset_index(drop=True)
        returns_df[sym] = rts
    returns_df = returns_df.dropna(axis=1, how="any")
    if returns_df.shape[1] < 2:
        return
    pca = PCA(n_components=3)
    pca.fit(returns_df.values)
    var_explained = pca.explained_variance_ratio_[0]
    if var_explained < 0.4:
        return
    top_loadings = pd.Series(pca.components_[0], index=returns_df.columns).abs()
    # Identify symbols loading > median loading
    median_load = top_loadings.median()
    high_load_syms = top_loadings[top_loadings > median_load].index.tolist()
    if not high_load_syms:
        return
    # Reduce those weights by 20%
    for sym in high_load_syms:
        old = ctx.portfolio_weights.get(sym, 0.0)
        ctx.portfolio_weights[sym] = round(old * 0.8, 4)
    # Re-normalize to sum to 1
    total = sum(ctx.portfolio_weights.values())
    if total > 0:
        for sym in ctx.portfolio_weights:
            ctx.portfolio_weights[sym] = round(ctx.portfolio_weights[sym] / total, 4)
    logger.info("PCA_ADJUSTMENT_APPLIED", extra={
        "var_explained": round(var_explained, 3),
        "adjusted": high_load_syms
    })

def daily_reset() -> None:
    """Reset daily counters and in-memory slippage logs."""
    global _slippage_log, _LOSS_STREAK
    _slippage_log.clear()
    _LOSS_STREAK = 0
    logger.info("DAILY_STATE_RESET")

# At top‐level, define retrain_meta_learner = None so load_or_retrain_daily can reference it safely
retrain_meta_learner = None

def load_or_retrain_daily(ctx: BotContext) -> Any:
    """
    1. Check RETRAIN_MARKER_FILE for last retrain date (YYYY-MM-DD).
    2. If missing or older than today, call retrain_meta_learner(ctx, symbols) and update marker.
    3. Then load the (new) model from MODEL_PATH.
    """
    today_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    marker = RETRAIN_MARKER_FILE

    need_to_retrain = True
    if os.path.isfile(marker):
        with open(marker, "r") as f:
            last_date = f.read().strip()
        if last_date == today_str:
            need_to_retrain = False

    if not os.path.exists(MODEL_PATH):
        logger.warning(
            "MODEL_PATH missing; forcing initial retrain.",
            extra={"path": MODEL_PATH},
        )
        need_to_retrain = True

    if need_to_retrain:
        if retrain_meta_learner is None:
            logger.warning("Daily retraining requested, but retrain_meta_learner is unavailable.")
        else:
            if not meta_lock.acquire(blocking=False):
                logger.warning("METALEARN_SKIPPED_LOCKED")
            else:
                try:
                    symbols = load_tickers(TICKERS_FILE)
                    logger.info(
                        f"▶ Starting meta-learner retraining for {today_str} on {len(symbols)} tickers..."
                    )
                    force_train = not os.path.exists(MODEL_PATH)
                    success = retrain_meta_learner(ctx, symbols, force=force_train)
                    if success:
                        try:
                            with open(marker, "w") as f:
                                f.write(today_str)
                        except Exception as e:
                            logger.warning(f"Failed to write retrain marker file: {e}")
                    else:
                        logger.warning("Retraining failed; continuing with existing model.")
                finally:
                    meta_lock.release()

    # Finally, load whichever model is at MODEL_PATH
    model = load_model(MODEL_PATH)
    if model is None:
        raise FileNotFoundError(f"Model not found at {MODEL_PATH} after retrain")
    return model

# ─── M. MAIN LOOP & SCHEDULER ─────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/health")
def health() -> str:
    return "OK", 200

def start_healthcheck() -> None:
    port = int(os.getenv("HEALTHCHECK_PORT", "8080"))
    try:
        app.run(host="0.0.0.0", port=port)
    except OSError as e:
        logger.warning(f"Healthcheck port {port} in use: {e}. Skipping health-endpoint.")

def run_all_trades_worker(model):
    global _last_fh_prefetch_date, _running
    if _running:
        logger.warning("RUN_ALL_TRADES_SKIPPED_OVERLAP")
        return
    _running = True
    try:
        # On each run, clear open orders and correct any position mismatches
        cancel_all_open_orders(ctx)
        audit_positions(ctx)
        now_utc = pd.Timestamp.utcnow()

        # Update SPY vol stats first
        compute_spy_vol_stats(ctx)

        logger.info("RUN_ALL_TRADES_START", extra={"timestamp": datetime.now(timezone.utc).isoformat()})

        candidates = load_tickers(TICKERS_FILE)
        today_date = date.today()
        if _last_fh_prefetch_date != today_date:
            _last_fh_prefetch_date = today_date
            # Bulk prefetch daily in batches of 10 to avoid rate limits
            all_syms = [s for s in candidates if s not in REGIME_SYMBOLS]
            for batch in chunked(all_syms, 10):
                try:
                    start_str = (today_date - timedelta(days=30)).isoformat()
                    end_str = today_date.isoformat()
                    bars = ctx.api.get_bars(
                        list(batch),
                        TimeFrame.Day,
                        start=start_str,
                        end=end_str,
                        limit=1000,
                        feed="iex"
                    ).df
                    for sym, df_sym in bars.groupby("symbol"):
                        df_df = df_sym.drop(columns=["symbol"], errors="ignore").copy()
                        df_df = df_df.rename(columns={
                            "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"
                        })
                        df_df.index = pd.to_datetime(df_df.index).tz_localize(None)
                        with cache_lock:
                            data_fetcher._daily_cache[sym] = df_df
                        daily_cache_hit.inc()
                    logger.info("PREFETCH_ALPACA_BATCH", extra={"batch": batch})
                except Exception as e:
                    logger.warning(f"[prefetch] Alpaca bulk failed for {batch}: {e} — falling back to Finnhub")
                    for sym in batch:
                        try:
                            df_sym = fh.fetch(sym, period="1mo", interval="1d")
                            if df_sym is not None and not df_sym.empty:
                                df_sym.index = pd.to_datetime(df_sym.index)
                                with cache_lock:
                                    data_fetcher._daily_cache[sym] = df_sym
                                daily_cache_hit.inc()
                                logger.info("PREFETCH_FINNHUB", extra={"symbol": sym})
                                continue
                        except Exception as e2:
                            logger.warning(f"[prefetch] Finnhub fetch failed for {sym}: {e2}")
                        dates = pd.date_range(start=(today_date - timedelta(days=30)), end=today_date, freq="D")
                        dummy = pd.DataFrame(index=dates, columns=["Open", "High", "Low", "Close", "Volume"])
                        dummy[["Open", "High", "Low", "Close"]] = np.nan
                        dummy["Volume"] = 0
                        with cache_lock:
                            data_fetcher._daily_cache[sym] = dummy
                        daily_cache_miss.inc()
                        logger.warning("DUMMY_DAILY_INSERTED", extra={"symbol": sym})
                pytime.sleep(1)  # avoid exceeding Alpaca rate

            # Ensure regime symbols seeded
            for sym in REGIME_SYMBOLS:
                data_fetcher.get_daily_df(ctx, sym)

        # Screen universe & compute weights
        tickers = screen_universe(candidates, ctx)
        logger.info("CANDIDATES_SCREENED", extra={"tickers": tickers})
        ctx.portfolio_weights = compute_portfolio_weights(tickers)
        if not tickers:
            logger.error("NO_TICKERS_TO_TRADE")
            return

        if check_halt_flag():
            logger.info("TRADING_HALTED_VIA_FLAG")
            return

        acct = ctx.api.get_account()
        current_cash = float(acct.cash)
        regime_ok = check_market_regime()

        futures = []
        for symbol in tickers:
            futures.append(executor.submit(_safe_trade, ctx, symbol, current_cash, model, regime_ok))

        # Wait for all trades to finish before allowing next run
        for f in as_completed(futures):
            try:
                f.result()
            except Exception:
                pass

        logger.info("RUN_ALL_TRADES_COMPLETE")
    finally:
        # Always reset running flag
        _running = False

def schedule_run_all_trades(model):
    Thread(target=run_all_trades_worker, args=(model,), daemon=True).start()

def initial_rebalance(ctx: BotContext, symbols: List[str]) -> None:
    acct = ctx.api.get_account()
    equity = float(acct.equity)
    if equity < PDT_EQUITY_THRESHOLD:
        logger.info("INITIAL_REBALANCE_SKIPPED_PDT", extra={"equity": equity})
        return
    cash = float(acct.cash)
    n = len(symbols)
    if n == 0 or cash <= 0:
        logger.info("INITIAL_REBALANCE_NO_SYMBOLS_OR_NO_CASH")
        return
    per_symbol = cash / n
    for sym in symbols:
        try:
            quote = ctx.api.get_latest_quote(sym)
            price = float(getattr(quote, "ask_price", 0) or 0)
            if price <= 0:
                logger.warning("INITIAL_REBALANCE_INVALID_PRICE", extra={"symbol": sym, "price": price})
                continue
            qty = int(per_symbol // price)
            if qty <= 0:
                continue
            logger.info("INITIAL_REBALANCE_BUY", extra={"symbol": sym, "qty": qty, "price": price})
            try:
                submit_order(ctx, sym, qty, "buy")
            except APIError as e:
                msg = str(e).lower()
                if "insufficient" in msg or "day trading buying power" in msg:
                    logger.warning("INITIAL_REBALANCE_SKIPPED_INSUFFICIENT", extra={"symbol": sym, "error": str(e)})
                else:
                    logger.exception("INITIAL_REBALANCE_ERROR", extra={"symbol": sym, "error": str(e)})
        except Exception as e:
            logger.exception("INITIAL_REBALANCE_FETCH_ERROR", extra={"symbol": sym, "error": str(e)})

if __name__ == "__main__":
    def _handle_term(signum, frame):
        logger.info("PROCESS_TERMINATION", extra={"signal": signum})
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)

    try:
        logger.info(">>> BOT __main__ ENTERED – starting up")
        
        # Try to start Prometheus metrics server; if port is already in use, log and continue
        try:
            start_http_server(9200)
        except OSError as e:
            logger.warning(f"Metrics server port 9200 already in use; skipping start_http_server: {e!r}")

        if RUN_HEALTH:
            Thread(target=start_healthcheck, daemon=True).start()

        # Daily jobs
        schedule.every().day.at("00:30").do(lambda: Thread(target=daily_summary, daemon=True).start())
        schedule.every().day.at("00:05").do(lambda: Thread(target=daily_reset, daemon=True).start())
        schedule.every().day.at("10:00").do(lambda: Thread(target=run_meta_learning_weight_optimizer, daemon=True).start())
        schedule.every().day.at("02:00").do(lambda: Thread(target=run_bayesian_meta_learning_optimizer, daemon=True).start())

        # ⮕ Only now import retrain_meta_learner, to avoid circular import / duplicated metrics
        try:
            from retrain import retrain_meta_learner as _tmp_retrain
            retrain_meta_learner = _tmp_retrain
        except ImportError:
            retrain_meta_learner = None
            logger.warning("retrain.py not found or retrain_meta_learner missing. Daily retraining disabled.")

        model = load_or_retrain_daily(ctx)
        logger.info("BOT_LAUNCHED")
        cancel_all_open_orders(ctx)
        audit_positions(ctx)

        # ─── WARM-CACHE SENTIMENT FOR ALL TICKERS ─────────────────────────────────────
        # This will prevent the initial burst of NewsAPI calls and 429s
        all_tickers = load_tickers(TICKERS_FILE)
        now_ts = pytime.time()
        with sentiment_lock:
            for t in all_tickers:
                _SENTIMENT_CACHE[t] = (now_ts, 0.0)

        # Initial rebalance (once)
        try:
            if not getattr(ctx, "_rebalance_done", False):
                universe = load_tickers(TICKERS_FILE)
                initial_rebalance(ctx, universe)
                ctx._rebalance_done = True
        except Exception as e:
            logger.warning(f"[REBALANCE] aborted due to error: {e}")

        # Recurring jobs
        schedule.every(1).minutes.do(lambda: schedule_run_all_trades(model))
        schedule.every(1).minutes.do(lambda: Thread(target=validate_open_orders, args=(ctx,), daemon=True).start())
        schedule.every(6).hours.do(lambda: Thread(target=update_signal_weights, daemon=True).start())

        # Scheduler loop
        while True:
            schedule.run_pending()
            pytime.sleep(1)

    except Exception as e:
        logger.exception(f"Fatal error in __main__: {e}")
        raise
