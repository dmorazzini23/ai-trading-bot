import warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:  # pragma: no cover - sklearn optional
    class InconsistentVersionWarning(UserWarning):
        pass

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import os
import config
from config import SHADOW_MODE

# Refresh environment variables on startup for reliability
config.reload_env()

# BOT_MODE must be defined before any classes that reference it
BOT_MODE = config.get_env("BOT_MODE", "balanced")
assert BOT_MODE is not None, "BOT_MODE must be set before using BotState"
from argparse import ArgumentParser

import logging
import logging.handlers
import csv
import json
import re
import time
import time as pytime

import random
import signal
import sys
import atexit
from datetime import datetime, date, time as dt_time, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Optional, Tuple, Dict, List, Any, Sequence
from threading import Semaphore, Lock, Thread
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from collections import deque

import numpy as np

# Set deterministic random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch

    torch.manual_seed(SEED)
except ImportError:
    pass

# Ensure numpy.NaN exists for pandas_ta compatibility
np.NaN = np.nan

import pandas as pd
import pandas_ta as ta
import pandas_market_calendars as mcal

warnings.filterwarnings("ignore", category=FutureWarning)

import requests
from bs4 import BeautifulSoup
from flask import Flask
import schedule
import portalocker
import yfinance as yf
from requests.exceptions import HTTPError
from alerts import send_slack_alert
from rebalancer import maybe_rebalance
from config import REBALANCE_INTERVAL_MIN

# Alpaca v3 SDK imports
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    QueryOrderStatus,
    OrderStatus,
)
from alpaca.trading.requests import (
    GetOrdersRequest,
    MarketOrderRequest,
    LimitOrderRequest,
)
from alpaca.trading.models import Order
from alpaca.common.exceptions import APIError

# Legacy import removed; using alpaca-py trading stream instead
from alpaca.trading.stream import TradingStream

# for paper trading
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
import threading
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.models import Quote
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.decomposition import PCA
import pickle
import joblib
from pipeline import model_pipeline
from utils import model_lock, safe_to_datetime
from metrics_logger import log_metrics

import sentry_sdk

import config

ALPACA_API_KEY = getattr(config, "ALPACA_API_KEY", None)
ALPACA_SECRET_KEY = getattr(config, "ALPACA_SECRET_KEY", None)
ALPACA_PAPER = getattr(config, "ALPACA_PAPER", None)
validate_alpaca_credentials = getattr(config, "validate_alpaca_credentials", None)
CONFIG_NEWS_API_KEY = getattr(config, "NEWS_API_KEY", None)
FINNHUB_API_KEY = getattr(config, "FINNHUB_API_KEY", None)
SENTRY_DSN = getattr(config, "SENTRY_DSN", None)
BOT_MODE_ENV = getattr(config, "BOT_MODE", BOT_MODE)
RUN_HEALTHCHECK = getattr(config, "RUN_HEALTHCHECK", None)


def _require_cfg(value, name):
    """Require a config value; sleep-and-retry in production."""
    if value:
        return value
    if BOT_MODE_ENV == "production":
        while not value:
            logger.critical(f"Missing {name}; retrying in 60s")
            time.sleep(60)
            config.reload_env()
            import importlib

            importlib.reload(config)
            value = getattr(config, name, None)
        return value
    raise RuntimeError(f"{name} must be defined in the configuration or environment")


ALPACA_API_KEY = _require_cfg(ALPACA_API_KEY, "ALPACA_API_KEY")
ALPACA_SECRET_KEY = _require_cfg(ALPACA_SECRET_KEY, "ALPACA_SECRET_KEY")
if not callable(validate_alpaca_credentials):
    raise RuntimeError("validate_alpaca_credentials not found in config")
BOT_MODE_ENV = _require_cfg(BOT_MODE_ENV, "BOT_MODE")
FINNHUB_API_KEY = _require_cfg(FINNHUB_API_KEY, "FINNHUB_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from prometheus_client import start_http_server, Counter, Gauge, Histogram
from finnhub import FinnhubAPIException
import pybreaker
try:
    from trade_execution import ExecutionEngine
except Exception:  # pragma: no cover - allow tests with stubbed module
    class ExecutionEngine:
        def __init__(self, *args, **kwargs):
            pass

try:
    from capital_scaling import CapitalScalingEngine
except Exception:  # pragma: no cover - allow tests with stubbed module
    class CapitalScalingEngine:
        def __init__(self, *args, **kwargs):
            pass
from data_fetcher import (
    finnhub_client,
    DataFetchError,
)
from strategy_allocator import StrategyAllocator
from risk_engine import RiskEngine
from utils import is_market_open as utils_market_open, portfolio_lock
from logger import get_logger
from strategies import MomentumStrategy, MeanReversionStrategy, TradeSignal

# Basic logger setup so early code can log before full configuration below
logger = get_logger(__name__)


def market_is_open(now: datetime | None = None) -> bool:
    """Return True if the market is currently open."""
    return utils_market_open(now)

# backward compatibility
is_market_open = market_is_open


def get_latest_close(df: pd.DataFrame) -> float:
    """
    Return the most recent closing price from df["close"].
    If df is None, empty, missing "close", or the last value is NaN or ≤0, return 0.0.
    """
    if df is None or df.empty:
        return 0.0
    if "close" in df.columns:
        last = df["close"].iloc[-1]
    else:
        return 0.0

    if pd.isna(last) or last <= 0:
        return 0.0
    return float(last)


def compute_time_range(minutes: int = 30) -> tuple[datetime, datetime]:
    """Return start and end timestamps for the last `minutes` minutes."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes)
    return start, end


def fetch_minute_df_safe(ctx: "BotContext", symbol: str) -> pd.DataFrame:
    """Return minute bars if market is open; otherwise an empty DataFrame."""
    if not market_is_open():
        logger.warning("MARKET_CLOSED_MINUTE_FETCH_SKIP", extra={"symbol": symbol})
        return pd.DataFrame()
    return ctx.data_fetcher.get_minute_df(ctx, symbol)


def cancel_all_open_orders(ctx: "BotContext") -> None:
    """
    On startup or each run, cancel every Alpaca order whose status is 'open'.
    """
    try:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        open_orders = ctx.api.get_orders(req)
        for od in open_orders:
            if getattr(od, "status", "").lower() == "open":
                try:
                    ctx.api.cancel_order_by_id(od.id)
                except Exception:
                    pass
    except Exception:
        pass


def reconcile_positions(ctx: "BotContext") -> None:
    """On startup, fetch all live positions and clear any in-memory stop/take targets for assets no longer held."""
    try:
        live_positions = {
            pos.symbol: int(pos.qty) for pos in ctx.api.get_all_positions()
        }
        with targets_lock:
            symbols_with_targets = list(ctx.stop_targets.keys()) + list(
                ctx.take_profit_targets.keys()
            )
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
    RetryError,
)
from ratelimit import limits, sleep_and_retry

import warnings

# ─── A. CONFIGURATION CONSTANTS ─────────────────────────────────────────────────
RUN_HEALTH = RUN_HEALTHCHECK == "1"

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
        logging.StreamHandler(),
        file_handler,
    ],
    force=True,
)
atexit.register(logging.shutdown)
logger = get_logger(__name__)
logging.getLogger("alpaca_trade_api").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# Sentry
sentry_sdk.init(
    dsn=SENTRY_DSN,
    traces_sample_rate=0.1,
    environment=BOT_MODE_ENV,
    default_integrations=False,  # FIXED: disable threading integration
)

# Suppress specific pandas_ta warnings
warnings.filterwarnings(
    "ignore", message=".*valid feature names.*", category=UserWarning
)

# ─── FINBERT SENTIMENT MODEL IMPORTS & FALLBACK ─────────────────────────────────
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    _FINBERT_TOKENIZER = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    _FINBERT_MODEL = AutoModelForSequenceClassification.from_pretrained(
        "yiyanghkust/finbert-tone"
    )
    _FINBERT_MODEL.eval()
    _HUGGINGFACE_AVAILABLE = True
    logger.info("FinBERT loaded successfully")
except Exception as e:
    _HUGGINGFACE_AVAILABLE = False
    _FINBERT_TOKENIZER = None
    _FINBERT_MODEL = None
    logger.warning(f"FinBERT load failed ({e}); falling back to neutral sentiment")

# Prometheus metrics
orders_total = Counter("bot_orders_total", "Total orders sent")
order_failures = Counter("bot_order_failures", "Order submission failures")
daily_drawdown = Gauge("bot_daily_drawdown", "Current daily drawdown fraction")
signals_evaluated = Counter("bot_signals_evaluated_total", "Total signals evaluated")
run_all_trades_duration = Histogram(
    "run_all_trades_duration_seconds", "Time spent in run_all_trades"
)
minute_cache_hit = Counter("bot_minute_cache_hits", "Minute bar cache hits")
minute_cache_miss = Counter("bot_minute_cache_misses", "Minute bar cache misses")
daily_cache_hit = Counter("bot_daily_cache_hits", "Daily bar cache hits")
daily_cache_miss = Counter("bot_daily_cache_misses", "Daily bar cache misses")
event_cooldown_hits = Counter("bot_event_cooldown_hits", "Event cooldown hits")
slippage_total = Counter("bot_slippage_total", "Cumulative slippage in cents")
slippage_count = Counter("bot_slippage_count", "Number of orders with slippage logged")
weekly_drawdown = Gauge("bot_weekly_drawdown", "Current weekly drawdown fraction")

DISASTER_DD_LIMIT = float(config.get_env("DISASTER_DD_LIMIT", "0.2"))

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def abspath(fname: str) -> str:
    return os.path.join(BASE_DIR, fname)


def atomic_joblib_dump(obj, path: str) -> None:
    """Safely write joblib file using atomic replace."""
    import tempfile

    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    os.close(fd)
    try:
        joblib.dump(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def atomic_pickle_dump(obj, path: str) -> None:
    """Safely pickle object to path with atomic replace."""
    import tempfile

    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(obj, f)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def get_git_hash() -> str:
    """Return current git commit short hash if available."""
    try:
        import subprocess

        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


TICKERS_FILE = abspath("tickers.csv")
TRADE_LOG_FILE = abspath("trades.csv")
SIGNAL_WEIGHTS_FILE = abspath("signal_weights.csv")
EQUITY_FILE = abspath("last_equity.txt")
PEAK_EQUITY_FILE = abspath("peak_equity.txt")
HALT_FLAG_PATH = abspath("halt.flag")
SLIPPAGE_LOG_FILE = abspath("slippage.csv")
REWARD_LOG_FILE = abspath("reward_log.csv")
FEATURE_PERF_FILE = abspath("feature_perf.csv")
INACTIVE_FEATURES_FILE = abspath("inactive_features.json")

# Hyperparameter files
HYPERPARAMS_FILE = abspath("hyperparams.json")
BEST_HYPERPARAMS_FILE = abspath("best_hyperparams.json")


def load_hyperparams() -> dict:
    """Load hyperparameters from best_hyperparams.json if present, else default."""
    path = (
        BEST_HYPERPARAMS_FILE
        if os.path.exists(BEST_HYPERPARAMS_FILE)
        else HYPERPARAMS_FILE
    )
    if not os.path.exists(path):
        logger.warning(f"Hyperparameter file {path} not found; using defaults")
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load hyperparameters from {path}: {e}")
        return {}


# <-- NEW: marker file for daily retraining -->
RETRAIN_MARKER_FILE = abspath("last_retrain.txt")

# Main meta‐learner path: this is where retrain.py will dump the new sklearn model each day.
MODEL_PATH = abspath(config.get_env("MODEL_PATH", "meta_model.pkl"))
MODEL_RF_PATH = abspath(config.get_env("MODEL_RF_PATH", "model_rf.pkl"))
MODEL_XGB_PATH = abspath(config.get_env("MODEL_XGB_PATH", "model_xgb.pkl"))
MODEL_LGB_PATH = abspath(config.get_env("MODEL_LGB_PATH", "model_lgb.pkl"))

REGIME_MODEL_PATH = abspath("regime_model.pkl")
# (We keep a separate meta‐model for signal‐weight learning, if you use Bayesian/Ridge, etc.)
META_MODEL_PATH = abspath("meta_model.pkl")


# Strategy mode
class BotMode:
    def __init__(self, mode: str = "balanced") -> None:
        self.mode = mode.lower()
        self.params = self.set_parameters()

    def set_parameters(self) -> dict[str, float]:
        if self.mode == "conservative":
            return {
                "KELLY_FRACTION": 0.3,
                "CONF_THRESHOLD": 0.8,
                "CONFIRMATION_COUNT": 3,
                "TAKE_PROFIT_FACTOR": 1.2,
                "DAILY_LOSS_LIMIT": 0.05,
                "CAPITAL_CAP": 0.05,
                "TRAILING_FACTOR": 1.5,
            }
        elif self.mode == "aggressive":
            return {
                "KELLY_FRACTION": 0.75,
                "CONF_THRESHOLD": 0.6,
                "CONFIRMATION_COUNT": 1,
                "TAKE_PROFIT_FACTOR": 2.2,
                "DAILY_LOSS_LIMIT": 0.1,
                "CAPITAL_CAP": 0.1,
                "TRAILING_FACTOR": 2.0,
            }
        else:  # balanced
            return {
                "KELLY_FRACTION": 0.6,
                "CONF_THRESHOLD": 0.75,
                "CONFIRMATION_COUNT": 2,
                "TAKE_PROFIT_FACTOR": 1.8,
                "DAILY_LOSS_LIMIT": 0.07,
                "CAPITAL_CAP": 0.08,
                "TRAILING_FACTOR": 1.2,
            }

    def get_config(self) -> dict[str, float]:
        return self.params


@dataclass
class BotState:
    loss_streak: int = 0
    streak_halt_until: Optional[datetime] = None
    day_start_equity: Optional[tuple[date, float]] = None
    week_start_equity: Optional[tuple[date, float]] = None
    last_drawdown: float = 0.0
    updates_halted: bool = False
    running: bool = False
    current_regime: str = "sideways"
    rolling_losses: list[float] = field(default_factory=list)
    mode_obj: BotMode = field(default_factory=lambda: BotMode(BOT_MODE))


state = BotState()
logger.info(f"Trading mode is set to '{state.mode_obj.mode}'")
params = state.mode_obj.get_config()
params.update(load_hyperparams())

# Other constants
NEWS_API_KEY = CONFIG_NEWS_API_KEY
TRAILING_FACTOR = params.get("TRAILING_FACTOR", 1.2)
SECONDARY_TRAIL_FACTOR = 1.0
TAKE_PROFIT_FACTOR = params.get("TAKE_PROFIT_FACTOR", 1.8)
SCALING_FACTOR = params.get("SCALING_FACTOR", 0.3)
ORDER_TYPE = "market"
LIMIT_ORDER_SLIPPAGE = params.get("LIMIT_ORDER_SLIPPAGE", 0.005)
MAX_POSITION_SIZE = 1000
SLICE_THRESHOLD = 50
POV_SLICE_PCT = params.get("POV_SLICE_PCT", 0.05)
DAILY_LOSS_LIMIT = params.get("DAILY_LOSS_LIMIT", 0.07)
MAX_PORTFOLIO_POSITIONS = int(config.get_env("MAX_PORTFOLIO_POSITIONS", 15))
CORRELATION_THRESHOLD = 0.60
SECTOR_EXPOSURE_CAP = float(config.get_env("SECTOR_EXPOSURE_CAP", "0.4"))
MAX_OPEN_POSITIONS = int(config.get_env("MAX_OPEN_POSITIONS", "10"))
WEEKLY_DRAWDOWN_LIMIT = float(config.get_env("WEEKLY_DRAWDOWN_LIMIT", "0.15"))
MARKET_OPEN = dt_time(6, 30)
MARKET_CLOSE = dt_time(13, 0)
VOLUME_THRESHOLD = int(config.get_env("VOLUME_THRESHOLD", "50000"))
ENTRY_START_OFFSET = timedelta(minutes=params.get("ENTRY_START_OFFSET_MIN", 30))
ENTRY_END_OFFSET = timedelta(minutes=params.get("ENTRY_END_OFFSET_MIN", 15))
REGIME_LOOKBACK = 14
REGIME_ATR_THRESHOLD = 20.0
RF_ESTIMATORS = 300
RF_MAX_DEPTH = 3
RF_MIN_SAMPLES_LEAF = 5
ATR_LENGTH = 10
CONF_THRESHOLD = params.get("CONF_THRESHOLD", 0.75)
CONFIRMATION_COUNT = params.get("CONFIRMATION_COUNT", 2)
CAPITAL_CAP = params.get("CAPITAL_CAP", 0.08)
DOLLAR_RISK_LIMIT = float(config.get_env("DOLLAR_RISK_LIMIT", "0.02"))
PACIFIC = ZoneInfo("America/Los_Angeles")
PDT_DAY_TRADE_LIMIT = params.get("PDT_DAY_TRADE_LIMIT", 3)
PDT_EQUITY_THRESHOLD = params.get("PDT_EQUITY_THRESHOLD", 25_000.0)
BUY_THRESHOLD = params.get("BUY_THRESHOLD", 0.2)
FINNHUB_RPM = int(config.get_env("FINNHUB_RPM", "60"))

# Regime symbols (makes SPY configurable)
REGIME_SYMBOLS = ["SPY"]

# ─── THREAD-SAFETY LOCKS & CIRCUIT BREAKER ─────────────────────────────────────
cache_lock = Lock()
targets_lock = Lock()
vol_lock = Lock()
sentiment_lock = Lock()
slippage_lock = Lock()
meta_lock = Lock()

breaker = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60)
executor = ThreadPoolExecutor(max_workers=4)

# EVENT cooldown
_LAST_EVENT_TS = {}
EVENT_COOLDOWN = 15.0  # seconds
REBALANCE_HOLD_SECONDS = 600  # skip selling shortly after rebalance buys

# Loss streak kill-switch (managed via BotState)

# Volatility stats (for SPY ATR mean/std)
_VOL_STATS = {"mean": None, "std": None, "last_update": None, "last": None}

# Slippage logs (in-memory for quick access)
_slippage_log: List[Tuple[str, float, float, datetime]] = (
    []
)  # (symbol, expected, actual, timestamp)
# Ensure persistent slippage log file exists
if not os.path.exists(SLIPPAGE_LOG_FILE):
    try:
        with open(SLIPPAGE_LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "symbol", "expected", "actual", "slippage_cents"]
            )
    except Exception as e:
        logger.warning(f"Could not create slippage log {SLIPPAGE_LOG_FILE}: {e}")

# Sector cache for portfolio exposure calculations
_SECTOR_CACHE: Dict[str, str] = {}


# ─── TYPED EXCEPTION ─────────────────────────────────────────────────────────
class DataFetchErrorLegacy(Exception):
    pass


# ─── B. CLIENTS & SINGLETONS ─────────────────────────────────────────────────


def ensure_alpaca_credentials() -> None:
    """Verify Alpaca credentials are present before starting."""
    validate_alpaca_credentials()


ensure_alpaca_credentials()


# Prometheus-safe account fetch
@breaker
def safe_alpaca_get_account(ctx: "BotContext"):
    return ctx.api.get_account()


# ─── C. HELPERS ────────────────────────────────────────────────────────────────
def chunked(iterable: Sequence, n: int):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def ttl_seconds() -> int:
    """Configurable TTL for minute-bar cache (default 60s)."""
    return int(config.get_env("MINUTE_CACHE_TTL", "60"))


def asset_class_for(symbol: str) -> str:
    """Very small heuristic to map tickers to asset classes."""
    sym = symbol.upper()
    if sym.endswith("USD") and len(sym) == 6:
        return "forex"
    if sym.startswith("BTC") or sym.startswith("ETH"):
        return "crypto"
    return "equity"


def compute_spy_vol_stats(ctx: "BotContext") -> None:
    """Compute daily ATR mean/std on SPY for the past 1 year."""
    today = date.today()
    with vol_lock:
        if _VOL_STATS["last_update"] == today:
            return

    df = ctx.data_fetcher.get_daily_df(ctx, REGIME_SYMBOLS[0])
    if df is None or len(df) < 252 + ATR_LENGTH:
        return True

    # Compute ATR series for last 252 trading days
    atr_series = ta.atr(df["high"], df["low"], df["close"], length=ATR_LENGTH).dropna()
    if len(atr_series) < 252:
        return True

    recent = atr_series.iloc[-252:]
    mean_val = float(recent.mean())
    std_val = float(recent.std())
    last_val = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0

    with vol_lock:
        _VOL_STATS["mean"] = mean_val
        _VOL_STATS["std"] = std_val
        _VOL_STATS["last_update"] = today
        _VOL_STATS["last"] = last_val

    logger.info(
        "SPY_VOL_STATS_UPDATED",
        extra={"mean": mean_val, "std": std_val, "atr": last_val},
    )


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

    atr_series = ta.atr(
        spy_df["high"], spy_df["low"], spy_df["close"], length=ATR_LENGTH
    )
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
class FinnhubFetcherLegacy:
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
        retry=retry_if_exception_type(Exception),
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
                logger.warning(f"[FH] no data for {sym}: status={resp.get('s')}")
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
            frames.append(df)

        if not frames:
            return pd.DataFrame()
        if len(frames) == 1:
            return frames[0]
        return pd.concat(frames, axis=1, keys=syms, names=["Symbol", "Field"])


_last_fh_prefetch_date: Optional[date] = None


@dataclass
class DataFetcher:
    def __post_init__(self):
        self._daily_cache: dict[str, Optional[pd.DataFrame]] = {}
        self._minute_cache: dict[str, Optional[pd.DataFrame]] = {}
        self._minute_timestamps: dict[str, datetime] = {}

    def get_daily_df(self, ctx: "BotContext", symbol: str) -> Optional[pd.DataFrame]:
        symbol = symbol.upper()
        now_utc = datetime.now(timezone.utc)
        end_ts = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        start_ts = end_ts - timedelta(days=5)

        with cache_lock:
            if symbol in self._daily_cache:
                daily_cache_hit.inc()
                return self._daily_cache[symbol]

        api_key = config.get_env("APCA_API_KEY_ID")
        api_secret = config.get_env("APCA_API_SECRET_KEY")
        if not api_key or not api_secret:
            raise RuntimeError(
                "APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set for data fetching"
            )
        client = StockHistoricalDataClient(api_key, api_secret)

        try:
            req = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Day,
                start=start_ts,
                end=end_ts,
            )
            bars = client.get_stock_bars(req).df
            if isinstance(bars.columns, pd.MultiIndex):
                bars = bars.xs(symbol, level=0, axis=1)
            else:
                bars = bars.drop(columns=["symbol"], errors="ignore")
            if len(bars.index) and isinstance(bars.index[0], tuple):
                idx_vals = [t[0] for t in bars.index]
            else:
                idx_vals = bars.index
            idx = safe_to_datetime(idx_vals)
            if idx is None:
                logger.warning(f"Invalid daily index for {symbol}; skipping")
                return None
            bars.index = idx
            df = bars.rename(columns=lambda c: c.lower()).drop(
                columns=["symbol"], errors="ignore"
            )
        except APIError as e:
            err_msg = str(e).lower()
            if "subscription does not permit querying recent sip data" in err_msg:
                logger.warning(
                    f"ALPACA SUBSCRIPTION ERROR for {symbol}: {repr(e)}"
                )
                logger.info(
                    f"ATTEMPTING IEX-DELAYERED DATA FOR {symbol}"
                )
                try:
                    req.feed = "iex"
                    df_iex = client.get_stock_bars(req).df
                    if isinstance(df_iex.columns, pd.MultiIndex):
                        df_iex = df_iex.xs(symbol, level=0, axis=1)
                    else:
                        df_iex = df_iex.drop(columns=["symbol"], errors="ignore")
                    if len(df_iex.index) and isinstance(df_iex.index[0], tuple):
                        idx_vals = [t[0] for t in df_iex.index]
                    else:
                        idx_vals = df_iex.index
                    idx = safe_to_datetime(idx_vals)
                    if idx is None:
                        logger.warning(f"Invalid IEX daily index for {symbol}; skipping")
                        return None
                    df_iex.index = idx
                    df = df_iex.rename(columns=lambda c: c.lower())
                except Exception as iex_err:
                    logger.warning(
                        f"ALPACA IEX ERROR for {symbol}: {repr(iex_err)}"
                    )
                    logger.info(
                        f"INSERTING DUMMY DAILY FOR {symbol} ON {end_ts.date().isoformat()}"
                    )
                    ts = pd.to_datetime(end_ts, utc=True, errors="coerce")
                    if ts is None:
                        ts = pd.Timestamp.utcnow()
                    dummy_date = ts
                    df = pd.DataFrame(
                        [
                            {
                                "open": 0.0,
                                "high": 0.0,
                                "low": 0.0,
                                "close": 0.0,
                                "volume": 0,
                            }
                        ],
                        index=[dummy_date],
                    )
            else:
                logger.warning(
                    f"ALPACA DAILY FETCH ERROR for {symbol}: {repr(e)}"
                )
                ts2 = pd.to_datetime(end_ts, utc=True, errors="coerce")
                if ts2 is None:
                    ts2 = pd.Timestamp.utcnow()
                dummy_date = ts2
                df = pd.DataFrame(
                    [{"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0}],
                    index=[dummy_date],
                )
        except Exception as e:
            logger.warning(
                f"ALPACA DAILY FETCH EXCEPTION for {symbol}: {repr(e)}"
            )
            ts = pd.to_datetime(end_ts, utc=True, errors="coerce")
            if ts is None:
                ts = pd.Timestamp.utcnow()
            dummy_date = ts
            df = pd.DataFrame(
                [{"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0}],
                index=[dummy_date],
            )

        with cache_lock:
            self._daily_cache[symbol] = df
        return df

    def get_minute_df(
        self, ctx: "BotContext", symbol: str, lookback_minutes: int = 30
    ) -> Optional[pd.DataFrame]:
        symbol = symbol.upper()
        now_utc = datetime.now(timezone.utc)
        last_closed_minute = now_utc.replace(second=0, microsecond=0) - timedelta(
            minutes=1
        )
        start_minute = last_closed_minute - timedelta(minutes=lookback_minutes)

        with cache_lock:
            last_ts = self._minute_timestamps.get(symbol)
            if last_ts and last_ts > now_utc - timedelta(seconds=ttl_seconds()):
                minute_cache_hit.inc()
                return self._minute_cache[symbol]

        minute_cache_miss.inc()
        api_key = config.get_env("APCA_API_KEY_ID")
        api_secret = config.get_env("APCA_API_SECRET_KEY")
        if not api_key or not api_secret:
            raise RuntimeError(
                "APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set for data fetching"
            )
        client = StockHistoricalDataClient(api_key, api_secret)

        try:
            req = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Minute,
                start=start_minute,
                end=last_closed_minute,
            )
            bars = client.get_stock_bars(req).df
            if isinstance(bars.columns, pd.MultiIndex):
                bars = bars.xs(symbol, level=0, axis=1)
            else:
                bars = bars.drop(columns=["symbol"], errors="ignore")
            if len(bars.index) and isinstance(bars.index[0], tuple):
                idx_vals = [t[0] for t in bars.index]
            else:
                idx_vals = bars.index
            idx = safe_to_datetime(idx_vals)
            if idx is None:
                logger.warning(f"Invalid minute index for {symbol}; skipping")
                return None
            bars.index = idx
            df = bars.rename(columns=lambda c: c.lower()).drop(
                columns=["symbol"], errors="ignore"
            )[["open", "high", "low", "close", "volume"]]
        except APIError as e:
            err_msg = str(e)
            if (
                "subscription does not permit querying recent sip data"
                in err_msg.lower()
            ):
                logger.warning(
                    f"ALPACA SUBSCRIPTION ERROR for {symbol}: {repr(e)}"
                )
                logger.info(
                    f"ATTEMPTING IEX-DELAYERED DATA FOR {symbol}"
                )
                try:
                    req.feed = "iex"
                    df_iex = client.get_stock_bars(req).df
                    if isinstance(df_iex.columns, pd.MultiIndex):
                        df_iex = df_iex.xs(symbol, level=0, axis=1)
                    else:
                        df_iex = df_iex.drop(columns=["symbol"], errors="ignore")
                    if len(df_iex.index) and isinstance(df_iex.index[0], tuple):
                        idx_vals = [t[0] for t in df_iex.index]
                    else:
                        idx_vals = df_iex.index
                    idx = safe_to_datetime(idx_vals)
                    if idx is None:
                        logger.warning(
                            f"Invalid IEX minute index for {symbol}; skipping"
                        )
                        df = pd.DataFrame()
                    else:
                        df_iex.index = idx
                        df = df_iex.rename(columns=lambda c: c.lower())[ 
                            ["open", "high", "low", "close", "volume"]
                        ]
                except Exception as iex_err:
                    logger.warning(
                        f"ALPACA IEX ERROR for {symbol}: {repr(iex_err)}"
                    )
                    logger.info(f"NO ALTERNATIVE MINUTE DATA FOR {symbol}")
                    df = pd.DataFrame()
            else:
                logger.warning(
                    f"ALPACA MINUTE FETCH ERROR for {symbol}: {repr(e)}"
                )
                df = pd.DataFrame()
        except Exception as e:
            logger.warning(
                f"ALPACA MINUTE FETCH ERROR for {symbol}: {repr(e)}"
            )
            df = pd.DataFrame()

        with cache_lock:
            self._minute_cache[symbol] = df
            self._minute_timestamps[symbol] = now_utc
        return df

    def get_historical_minute(
        self,
        ctx: "BotContext",  # ← still needs ctx here, per retrain.py
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch all minute bars for `symbol` between start_date and end_date (inclusive),
        by calling Alpaca’s get_bars for each calendar day. Returns a DataFrame
        indexed by naive Timestamps, or None if no data was returned at all.
        """
        all_days: list[pd.DataFrame] = []
        current_day = start_date

        while current_day <= end_date:
            day_start = datetime.combine(current_day, dt_time.min, timezone.utc)
            day_end = datetime.combine(current_day, dt_time.max, timezone.utc)
            if isinstance(day_start, tuple):
                day_start, _tmp = day_start
            if isinstance(day_end, tuple):
                _, day_end = day_end

            try:
                bars_req = StockBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=TimeFrame.Minute,
                    start=day_start,
                    end=day_end,
                    limit=10000,
                    feed="iex",
                )
                try:
                    bars_day = ctx.data_client.get_stock_bars(bars_req).df
                except APIError as e:
                    logger.warning(
                        f"[historic_minute] SIP failed for {symbol} {day_start}-{day_end}: {e}; retrying with IEX"
                    )
                    bars_req.feed = "iex"
                    bars_day = ctx.data_client.get_stock_bars(bars_req).df
                if isinstance(bars_day.columns, pd.MultiIndex):
                    bars_day = bars_day.xs(symbol, level=0, axis=1)
                else:
                    bars_day = bars_day.drop(columns=["symbol"], errors="ignore")
            except Exception as e:
                logger.warning(
                    f"[historic_minute] failed for {symbol} {day_start}-{day_end}: {e}"
                )
                bars_day = None

            if bars_day is not None and not bars_day.empty:
                if "symbol" in bars_day.columns:
                    bars_day = bars_day.drop(columns=["symbol"], errors="ignore")

                idx = safe_to_datetime(bars_day.index)
                if idx is None:
                    logger.warning(
                        f"Invalid minute index for {symbol}; skipping day {day_start}"
                    )
                    bars_day = None
                else:
                    bars_day.index = idx
                    bars_day = bars_day.rename(columns=lambda c: c.lower())[
                        ["open", "high", "low", "close", "volume"]
                    ]
                    all_days.append(bars_day)

            current_day += timedelta(days=1)

        if not all_days:
            return None

        combined = pd.concat(all_days, axis=0)
        combined = combined[~combined.index.duplicated(keep="first")]
        combined = combined.sort_index()
        return combined


# Helper to prefetch daily data in bulk with Alpaca, handling SIP subscription
# issues and falling back to IEX delayed feed per symbol if needed.
def prefetch_daily_data(
    symbols: List[str], start_date: date, end_date: date
) -> Dict[str, pd.DataFrame]:
    alpaca_key = config.get_env("APCA_API_KEY_ID")
    alpaca_secret = config.get_env("APCA_API_SECRET_KEY")
    if not alpaca_key or not alpaca_secret:
        raise RuntimeError(
            "APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set for data fetching"
        )
    client = StockHistoricalDataClient(alpaca_key, alpaca_secret)

    try:
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date,
        )
        bars = client.get_stock_bars(req).df
        if isinstance(bars.columns, pd.MultiIndex):
            grouped_raw = {
                sym: bars.xs(sym, level=0, axis=1)
                for sym in symbols
                if sym in bars.columns.get_level_values(0)
            }
        else:
            grouped_raw = {sym: df for sym, df in bars.groupby("symbol")}
        grouped = {}
        for sym, df in grouped_raw.items():
            df = df.drop(columns=["symbol"], errors="ignore")
            idx = safe_to_datetime(df.index)
            if idx is None:
                logger.warning(f"Invalid bulk index for {sym}; skipping")
                continue
            df.index = idx
            df = df.rename(columns=lambda c: c.lower())
            grouped[sym] = df
        return grouped
    except APIError as e:
        err_msg = str(e).lower()
        if "subscription does not permit querying recent sip data" in err_msg:
            logger.warning(
                f"ALPACA SUBSCRIPTION ERROR in bulk for {symbols}: {repr(e)}"
            )
            logger.info(f"ATTEMPTING IEX-DELAYERED BULK FETCH FOR {symbols}")
            try:
                req.feed = "iex"
                bars_iex = client.get_stock_bars(req).df
                if isinstance(bars_iex.columns, pd.MultiIndex):
                    grouped_raw = {
                        sym: bars_iex.xs(sym, level=0, axis=1)
                        for sym in symbols
                        if sym in bars_iex.columns.get_level_values(0)
                    }
                else:
                    grouped_raw = {sym: df for sym, df in bars_iex.groupby("symbol")}
                grouped = {}
                for sym, df in grouped_raw.items():
                    df = df.drop(columns=["symbol"], errors="ignore")
                    idx = safe_to_datetime(df.index)
                    if idx is None:
                        logger.warning(f"Invalid IEX bulk index for {sym}; skipping")
                        continue
                    df.index = idx
                    df = df.rename(columns=lambda c: c.lower())
                    grouped[sym] = df
                return grouped
            except Exception as iex_err:
                logger.warning(
                    f"ALPACA IEX BULK ERROR for {symbols}: {repr(iex_err)}"
                )
                daily_dict = {}
                for sym in symbols:
                    try:
                        req_sym = StockBarsRequest(
                            symbol_or_symbols=[sym],
                            timeframe=TimeFrame.Day,
                            start=start_date,
                            end=end_date,
                            feed="iex",
                        )
                        df_sym = client.get_stock_bars(req_sym).df
                        df_sym = df_sym.drop(columns=["symbol"], errors="ignore")
                        idx = safe_to_datetime(df_sym.index)
                        if idx is None:
                            logger.warning(
                                f"Invalid fallback bulk index for {sym}; skipping"
                            )
                            continue
                        df_sym.index = idx
                        df_sym = df_sym.rename(columns=lambda c: c.lower())
                        daily_dict[sym] = df_sym
                    except Exception as indiv_err:
                        logger.warning(
                            f"ALPACA IEX ERROR for {sym}: {repr(indiv_err)}"
                        )
                        logger.info(
                            f"INSERTING DUMMY DAILY FOR {sym} ON {end_date.isoformat()}"
                        )
                        tsd = pd.to_datetime(end_date, utc=True, errors="coerce")
                        if tsd is None:
                            tsd = pd.Timestamp.utcnow()
                        dummy_date = tsd
                        dummy_df = pd.DataFrame(
                            [
                                {
                                    "open": 0.0,
                                    "high": 0.0,
                                    "low": 0.0,
                                    "close": 0.0,
                                    "volume": 0,
                                }
                            ],
                            index=[dummy_date],
                        )
                        daily_dict[sym] = dummy_df
                return daily_dict
        else:
            logger.warning(
                f"ALPACA BULK FETCH UNKNOWN ERROR for {symbols}: {repr(e)}"
            )
            daily_dict = {}
            for sym in symbols:
                t2 = pd.to_datetime(end_date, utc=True, errors="coerce")
                if t2 is None:
                    t2 = pd.Timestamp.utcnow()
                dummy_date = t2
                dummy_df = pd.DataFrame(
                    [{"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0}],
                    index=[dummy_date],
                )
                daily_dict[sym] = dummy_df
            return daily_dict
    except Exception as e:
        logger.warning(
            f"ALPACA BULK FETCH EXCEPTION for {symbols}: {repr(e)}"
        )
        daily_dict = {}
        for sym in symbols:
            t3 = pd.to_datetime(end_date, utc=True, errors="coerce")
            if t3 is None:
                t3 = pd.Timestamp.utcnow()
            dummy_date = t3
            dummy_df = pd.DataFrame(
                [{"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0}],
                index=[dummy_date],
            )
            daily_dict[sym] = dummy_df
        return daily_dict


# ─── E. TRADE LOGGER ───────────────────────────────────────────────────────────
class TradeLogger:
    def __init__(self, path: str = TRADE_LOG_FILE) -> None:
        self.path = path
        if not os.path.exists(path):
            with portalocker.Lock(path, "w", timeout=5) as f:
                csv.writer(f).writerow(
                    [
                        "symbol",
                        "entry_time",
                        "entry_price",
                        "exit_time",
                        "exit_price",
                        "qty",
                        "side",
                        "strategy",
                        "classification",
                        "signal_tags",
                        "confidence",
                        "reward",
                    ]
                )
        if not os.path.exists(REWARD_LOG_FILE):
            try:
                with open(REWARD_LOG_FILE, "w", newline="") as rf:
                    csv.writer(rf).writerow(
                        [
                            "timestamp",
                            "symbol",
                            "reward",
                            "pnl",
                            "confidence",
                            "band",
                        ]
                    )
            except Exception as e:
                logger.warning(f"Failed to create reward log: {e}")

    def log_entry(
        self,
        symbol: str,
        price: float,
        qty: int,
        side: str,
        strategy: str,
        signal_tags: str = "",
        confidence: float = 0.0,
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with portalocker.Lock(self.path, "a", timeout=5) as f:
            csv.writer(f).writerow(
                [
                    symbol,
                    now_iso,
                    price,
                    "",
                    "",
                    qty,
                    side,
                    strategy,
                    "",
                    signal_tags,
                    confidence,
                    "",
                ]
            )

    def log_exit(self, state: BotState, symbol: str, exit_price: float) -> None:
        with portalocker.Lock(self.path, "r+", timeout=5) as f:
            rows = list(csv.reader(f))
            header, data = rows[0], rows[1:]
            pnl = 0.0
            conf = 0.0
            for row in data:
                if row[0] == symbol and row[3] == "":
                    entry_t = datetime.fromisoformat(row[1])
                    days = (datetime.now(timezone.utc) - entry_t).days
                    cls = (
                        "day_trade"
                        if days == 0
                        else "swing_trade" if days < 5 else "long_trade"
                    )
                    row[3], row[4], row[8] = (
                        datetime.now(timezone.utc).isoformat(),
                        exit_price,
                        cls,
                    )
                    # Compute PnL
                    entry_price = float(row[2])
                    pnl = (exit_price - entry_price) * (1 if row[6] == "buy" else -1)
                    if len(row) >= 11:
                        try:
                            conf = float(row[10])
                        except Exception:
                            conf = 0.0
                    if len(row) >= 12:
                        row[11] = pnl * conf
                    else:
                        row.append(conf)
                        row.append(pnl * conf)
                    break
            f.seek(0)
            f.truncate()
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(data)

        # log reward
        try:
            with open(REWARD_LOG_FILE, "a", newline="") as rf:
                csv.writer(rf).writerow(
                    [
                        datetime.now(timezone.utc).isoformat(),
                        symbol,
                        pnl * conf,
                        pnl,
                        conf,
                        ctx.capital_band,
                    ]
                )
        except Exception:
            pass

        # Update streak-based kill-switch
        if pnl < 0:
            state.loss_streak += 1
        else:
            state.loss_streak = 0
        if state.loss_streak >= 3:
            state.streak_halt_until = datetime.now(PACIFIC) + timedelta(minutes=60)
            logger.warning(
                "STREAK_HALT_TRIGGERED",
                extra={
                    "loss_streak": state.loss_streak,
                    "halt_until": state.streak_halt_until,
                },
            )


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
        remote = {p.symbol: int(p.qty) for p in ctx.api.get_all_positions()}
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
                    req = MarketOrderRequest(
                        symbol=sym,
                        qty=diff,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                    )
                    safe_submit_order(ctx.api, req)
                except Exception:
                    pass
            else:
                # Broker has fewer shares than local: buy back the missing shares
                try:
                    req = MarketOrderRequest(
                        symbol=sym,
                        qty=abs(diff),
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY,
                    )
                    safe_submit_order(ctx.api, req)
                except Exception:
                    pass

    # 4) For any symbol in local that is not in remote, submit order matching the local side
    for sym, lq in local.items():
        if sym not in remote:
            try:
                side = OrderSide.BUY if lq > 0 else OrderSide.SELL
                req = MarketOrderRequest(
                    symbol=sym, qty=abs(lq), side=side, time_in_force=TimeInForce.DAY
                )
                safe_submit_order(ctx.api, req)
            except Exception:
                pass


def validate_open_orders(ctx: "BotContext") -> None:
    try:
        open_orders = ctx.api.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
    except Exception:
        return

    now = datetime.now(timezone.utc)
    for od in open_orders:
        created = pd.to_datetime(getattr(od, "created_at", now))
        age = (now - created).total_seconds() / 60.0

        if age > 5 and getattr(od, "status", "").lower() in {"new", "accepted"}:
            try:
                ctx.api.cancel_order_by_id(od.id)
                qty = int(getattr(od, "qty", 0))
                side = getattr(od, "side", "")
                if qty > 0 and side in {"buy", "sell"}:
                    req = MarketOrderRequest(
                        symbol=od.symbol,
                        qty=qty,
                        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                    )
                    safe_submit_order(ctx.api, req)
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
            return -1, 0.0, "momentum"
        try:
            df["momentum"] = df["close"].pct_change(self.momentum_lookback)
            val = df["momentum"].iloc[-1]
            s = 1 if val > 0 else -1 if val < 0 else -1
            w = min(abs(val) * 10, 1.0)
            return s, w, "momentum"
        except Exception:
            logger.exception("Error in signal_momentum")
            return -1, 0.0, "momentum"

    def signal_mean_reversion(
        self, df: pd.DataFrame, model=None
    ) -> Tuple[int, float, str]:
        if df is None or len(df) < self.mean_rev_lookback:
            return -1, 0.0, "mean_reversion"
        try:
            ma = df["close"].rolling(self.mean_rev_lookback).mean()
            sd = df["close"].rolling(self.mean_rev_lookback).std()
            df["zscore"] = (df["close"] - ma) / sd
            val = df["zscore"].iloc[-1]
            s = (
                -1
                if val > self.mean_rev_zscore_threshold
                else 1 if val < -self.mean_rev_zscore_threshold else -1
            )
            w = min(abs(val) / 3, 1.0)
            return s, w, "mean_reversion"
        except Exception:
            logger.exception("Error in signal_mean_reversion")
            return -1, 0.0, "mean_reversion"

    def signal_stochrsi(self, df: pd.DataFrame, model=None) -> Tuple[int, float, str]:
        if df is None or "stochrsi" not in df or df["stochrsi"].dropna().empty:
            return -1, 0.0, "stochrsi"
        try:
            val = df["stochrsi"].iloc[-1]
            s = 1 if val < 0.2 else -1 if val > 0.8 else -1
            return s, 0.3, "stochrsi"
        except Exception:
            logger.exception("Error in signal_stochrsi")
            return -1, 0.0, "stochrsi"

    def signal_obv(self, df: pd.DataFrame, model=None) -> Tuple[int, float, str]:
        if df is None or len(df) < 6:
            return -1, 0.0, "obv"
        try:
            obv = pd.Series(ta.obv(df["close"], df["volume"]).values)
            if len(obv) < 5:
                return -1, 0.0, "obv"
            slope = np.polyfit(range(5), obv.tail(5), 1)[0]
            s = 1 if slope > 0 else -1 if slope < 0 else -1
            w = min(abs(slope) / 1e6, 1.0)
            return s, w, "obv"
        except Exception:
            logger.exception("Error in signal_obv")
            return -1, 0.0, "obv"

    def signal_vsa(self, df: pd.DataFrame, model=None) -> Tuple[int, float, str]:
        if df is None or len(df) < 20:
            return -1, 0.0, "vsa"
        try:
            body = abs(df["close"] - df["open"])
            vsa = df["volume"] * body
            score = vsa.iloc[-1]
            roll = vsa.rolling(20).mean()
            avg = roll.iloc[-1] if not roll.empty else 0.0
            s = (
                1
                if df["close"].iloc[-1] > df["open"].iloc[-1]
                else -1 if df["close"].iloc[-1] < df["open"].iloc[-1] else -1
            )
            w = min(score / avg, 1.0)
            return s, w, "vsa"
        except Exception:
            logger.exception("Error in signal_vsa")
            return -1, 0.0, "vsa"

    def signal_ml(self, df: pd.DataFrame, model: Any = None) -> Tuple[int, float, str]:
        """Machine learning prediction signal with probability logging."""
        try:
            if hasattr(model, "feature_names_in_"):
                feat = list(model.feature_names_in_)
            else:
                feat = ["rsi", "macd", "atr", "vwap", "sma_50", "sma_200"]

            X = df[feat].iloc[-1].values.reshape(1, -1)
            try:
                pred = model.predict(X)[0]
            except ValueError as e:
                logger.error(f"signal_ml failed: {e}")
                return -1, 0.0, "ml"
            proba = float(model.predict_proba(X)[0][pred])
            s = 1 if pred == 1 else -1
            logger.info(
                "ML_SIGNAL", extra={"prediction": int(pred), "probability": proba}
            )
            return s, proba, "ml"
        except Exception as e:
            logger.exception(f"signal_ml failed: {e}")
            return -1, 0.0, "ml"

    def signal_sentiment(
        self, ctx: "BotContext", ticker: str, df: pd.DataFrame = None, model: Any = None
    ) -> Tuple[int, float, str]:
        """
        Only fetch sentiment if price has moved > PRICE_TTL_PCT; otherwise, return cached/neutral.
        """
        if df is None or df.empty:
            return -1, 0.0, "sentiment"

        latest_close = float(get_latest_close(df))
        with sentiment_lock:
            prev_close = _LAST_PRICE.get(ticker, None)

        # If price hasn’t moved enough, return cached or neutral
        if (
            prev_close is not None
            and abs(latest_close - prev_close) / prev_close < PRICE_TTL_PCT
        ):
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
        return s, weight, "sentiment"

    def signal_regime(
        self, state: BotState, df: pd.DataFrame, model=None
    ) -> Tuple[int, float, str]:
        ok = check_market_regime(state)
        s = 1 if ok else -1
        return s, 1.0, "regime"

    def load_signal_weights(self) -> dict[str, float]:
        if not os.path.exists(SIGNAL_WEIGHTS_FILE):
            return {}
        df = pd.read_csv(SIGNAL_WEIGHTS_FILE)
        return {row["signal"]: row["weight"] for _, row in df.iterrows()}

    def evaluate(
        self,
        ctx: "BotContext",
        state: BotState,
        df: pd.DataFrame,
        ticker: str,
        model: Any,
    ) -> Tuple[int, float, str]:
        """
        Evaluate all signals, allowing short (s=-1) and long (s=1) components.
        Returns (final_signal, confidence ∈ [0,1], concatenated labels).
        """
        signals: List[Tuple[int, float, str]] = []
        allowed_tags = set(load_global_signal_performance() or [])
        weights = self.load_signal_weights()

        # Track total signals evaluated
        signals_evaluated.inc()

        # simple moving averages
        df["sma_50"] = df["close"].rolling(window=50).mean()
        df["sma_200"] = df["close"].rolling(window=200).mean()

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
                elif fn == self.signal_regime:
                    s, w, lab = fn(state, df, model)
                else:
                    s, w, lab = fn(df, model)
                if allowed_tags and lab not in allowed_tags:
                    continue
                if s in (-1, 1):
                    weight = weights.get(lab, w)
                    regime_adj = {
                        "trending": {"momentum": 1.2, "mean_reversion": 0.8},
                        "mean_reversion": {"momentum": 0.8, "mean_reversion": 1.2},
                        "high_volatility": {"sentiment": 1.1},
                        "sideways": {"momentum": 0.9, "mean_reversion": 1.1},
                    }
                    if (
                        state.current_regime in regime_adj
                        and lab in regime_adj[state.current_regime]
                    ):
                        weight *= regime_adj[state.current_regime][lab]
                    signals.append((s, weight, lab))
            except Exception:
                continue

        if not signals:
            return -1, 0.0, "no_signal"

        self.last_components = signals

        ml_prob = next((w for s, w, l in signals if l == "ml"), 0.5)
        adj = []
        for s, w, l in signals:
            if l != "ml":
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
    # Trading client using the new Alpaca SDK
    api: TradingClient
    # Separate market data client
    data_client: StockHistoricalDataClient
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
    capital_scaler: CapitalScalingEngine
    adv_target_pct: float
    max_position_dollars: float
    params: dict
    sector_cap: float = SECTOR_EXPOSURE_CAP
    correlation_limit: float = CORRELATION_THRESHOLD
    capital_band: str = "small"
    confirmation_count: dict[str, int] = field(default_factory=dict)
    trailing_extremes: dict[str, float] = field(default_factory=dict)
    take_profit_targets: dict[str, float] = field(default_factory=dict)
    stop_targets: dict[str, float] = field(default_factory=dict)
    portfolio_weights: dict[str, float] = field(default_factory=dict)
    tickers: List[str] = field(default_factory=list)
    rebalance_buys: dict[str, datetime] = field(default_factory=dict)
    risk_engine: RiskEngine | None = None
    allocator: StrategyAllocator | None = None
    strategies: List[Any] = field(default_factory=list)
    execution_engine: ExecutionEngine | None = None


data_fetcher = DataFetcher()
signal_manager = SignalManager()
trade_logger = TradeLogger()
risk_engine = RiskEngine()
allocator = StrategyAllocator()
strategies = [MomentumStrategy(), MeanReversionStrategy()]
API_KEY = ALPACA_API_KEY
SECRET_KEY = ALPACA_SECRET_KEY
BASE_URL = ALPACA_BASE_URL
paper = ALPACA_PAPER
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=paper)
# alias get_order for v2 SDK differences
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# WebSocket for order status updates
# use the new Stream class; explicitly set feed and base_url

# Create a trading stream for order status updates
stream = TradingStream(
    API_KEY,
    SECRET_KEY,
    paper=True,
)


async def on_trade_update(event):
    """Handle order status updates from the Alpaca stream."""
    logger.info(f"Trade update for {event.order['symbol']}: {event.order['status']}")


stream.subscribe_trade_updates(on_trade_update)
ctx = BotContext(
    api=trading_client,
    data_client=data_client,
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
    capital_scaler=CapitalScalingEngine(),
    adv_target_pct=0.002,
    max_position_dollars=10_000,
    params=params,
    confirmation_count={},
    trailing_extremes={},
    take_profit_targets={},
    stop_targets={},
    portfolio_weights={},
    rebalance_buys={},
    risk_engine=risk_engine,
    allocator=allocator,
    strategies=strategies,
)
exec_engine = ExecutionEngine(
    ctx,
    slippage_total=slippage_total,
    slippage_count=slippage_count,
    orders_total=orders_total,
)
ctx.execution_engine = exec_engine

try:
    equity_init = float(ctx.api.get_account().equity)
except Exception:
    equity_init = 0.0
ctx.capital_scaler.update(ctx, equity_init)

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
        logger.warning(
            f"No NYSE market schedule for {ts.date()}; skipping market open/close check."
        )
        return False
    return (
        schedule_today.market_open.iloc[0] <= ts <= schedule_today.market_close.iloc[0]
    )


# ─── I. SENTIMENT & EVENTS ────────────────────────────────────────────────────
@sleep_and_retry
@limits(calls=30, period=60)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(requests.RequestException),
)
def get_sec_headlines(ctx: BotContext, ticker: str) -> str:
    with ctx.sem:
        r = requests.get(
            f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
            f"&CIK={ticker}&type=8-K&count=5",
            headers={"User-Agent": "AI Trading Bot"},
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
    retry=retry_if_exception_type((requests.RequestException, DataFetchError)),
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
            logger.warning(
                f"fetch_sentiment({ticker}) rate-limited → returning neutral 0.0"
            )
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
            logger.warning(
                f"[predict_text_sentiment] FinBERT inference failed ({e}); returning neutral"
            )
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
        filings.append(
            {
                "date": fdate,
                "type": ("buy" if "purchase" in txn_type else "sell"),
                "dollar_amount": amt,
            }
        )
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
    except HTTPError:
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
            if "Value" not in cal.index or col not in cal.columns:
                continue
            raw = cal.at["Value", col]
            if isinstance(raw, (list, tuple)):
                raw = raw[0]
            dates.append(pd.to_datetime(raw))
    except Exception:
        logger.debug(
            f"[Events] Malformed calendar for {symbol}, columns={getattr(cal, 'columns', None)}"
        )
        return False
    today_ts = pd.Timestamp.now().normalize()
    cutoff = today_ts + pd.Timedelta(days=days)
    return any(today_ts <= d <= cutoff for d in dates)


# ─── J. RISK & GUARDS ─────────────────────────────────────────────────────────


@sleep_and_retry
@limits(calls=200, period=60)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type(APIError),
)
def check_daily_loss(ctx: BotContext, state: BotState) -> bool:
    acct = safe_alpaca_get_account(ctx)
    equity = float(acct.equity)
    today_date = date.today()
    limit = params.get("DAILY_LOSS_LIMIT", 0.07)

    if state.day_start_equity is None or state.day_start_equity[0] != today_date:
        if state.last_drawdown >= 0.05:
            limit = 0.03
        state.last_drawdown = (
            (state.day_start_equity[1] - equity) / state.day_start_equity[1]
            if state.day_start_equity
            else 0.0
        )
        state.day_start_equity = (today_date, equity)
        daily_drawdown.set(0.0)
        return False

    loss = (state.day_start_equity[1] - equity) / state.day_start_equity[1]
    daily_drawdown.set(loss)
    if loss > 0.05:
        sentry_sdk.capture_message(f"[WARNING] Daily drawdown = {loss:.2%}")
    return loss >= limit


def check_weekly_loss(ctx: BotContext, state: BotState) -> bool:
    """Weekly portfolio drawdown guard."""
    acct = safe_alpaca_get_account(ctx)
    equity = float(acct.equity)
    today_date = date.today()
    week_start = today_date - timedelta(days=today_date.weekday())

    if state.week_start_equity is None or state.week_start_equity[0] != week_start:
        state.week_start_equity = (week_start, equity)
        weekly_drawdown.set(0.0)
        return False

    loss = (state.week_start_equity[1] - equity) / state.week_start_equity[1]
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
        (df["entry_date"].isin(bdays))
        & (df["exit_date"].isin(bdays))
        & (df["entry_date"] == df["exit_date"])
    )
    return int(mask.sum())


@sleep_and_retry
@limits(calls=200, period=60)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type(APIError),
)
def check_pdt_rule(ctx: BotContext) -> bool:
    acct = safe_alpaca_get_account(ctx)
    equity = float(acct.equity)

    api_day_trades = getattr(acct, "pattern_day_trades", None) or getattr(
        acct, "pattern_day_trades_count", None
    )
    api_buying_pw = getattr(acct, "daytrade_buying_power", None) or getattr(
        acct, "day_trade_buying_power", None
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
    try:
        with open(HALT_FLAG_PATH, "r") as f:
            content = f.read().strip()
    except Exception:
        content = ""
    max_age = (
        timedelta(hours=24) if content.startswith("DISASTER") else timedelta(hours=1)
    )
    return age < max_age


def too_many_positions(ctx: BotContext) -> bool:
    try:
        return len(ctx.api.get_all_positions()) >= MAX_PORTFOLIO_POSITIONS
    except Exception:
        logger.warning("[too_many_positions] Could not fetch positions")
        return False


def too_correlated(ctx: BotContext, sym: str) -> bool:
    if not os.path.exists(TRADE_LOG_FILE):
        return False
    df = pd.read_csv(TRADE_LOG_FILE)
    if "exit_time" not in df.columns or "symbol" not in df.columns:
        return False
    open_syms = df.loc[df.exit_time == "", "symbol"].unique().tolist() + [sym]
    rets: Dict[str, pd.Series] = {}
    for s in open_syms:
        d = ctx.data_fetcher.get_daily_df(ctx, s)
        if d is None or d.empty:
            continue
        # Handle DataFrame with MultiIndex columns (symbol, field) or single-level
        if isinstance(d.columns, pd.MultiIndex):
            if (s, "close") in d.columns:
                series = d[(s, "close")].pct_change().dropna()
            else:
                continue
        else:
            series = d["close"].pct_change().dropna()
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
    limit = getattr(ctx, "correlation_limit", CORRELATION_THRESHOLD)
    return avg_corr > limit


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
        positions = ctx.api.get_all_positions()
    except Exception:
        return {}
    try:
        total = float(ctx.api.get_account().portfolio_value)
    except Exception:
        total = 0.0
    exposure: Dict[str, float] = {}
    for pos in positions:
        qty = abs(int(getattr(pos, "qty", 0)))
        price = float(
            getattr(pos, "current_price", 0) or getattr(pos, "avg_entry_price", 0) or 0
        )
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
    cap = getattr(ctx, "sector_cap", SECTOR_EXPOSURE_CAP)
    return projected <= cap


# ─── K. SIZING & EXECUTION HELPERS ─────────────────────────────────────────────
def is_within_entry_window(ctx: BotContext, state: BotState) -> bool:
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
    if state.streak_halt_until and datetime.now(PACIFIC) < state.streak_halt_until:
        logger.info("SKIP_STREAK_HALT", extra={"until": state.streak_halt_until})
        return False
    return True


def scaled_atr_stop(
    entry_price: float,
    atr: float,
    now: datetime,
    market_open: datetime,
    market_close: datetime,
    max_factor: float = 2.0,
    min_factor: float = 0.5,
) -> Tuple[float, float]:
    total = (market_close - market_open).total_seconds()
    elapsed = (now - market_open).total_seconds()
    α = max(0, min(1, 1 - elapsed / total))
    factor = min_factor + α * (max_factor - min_factor)
    stop = entry_price - factor * atr
    take = entry_price + factor * atr
    return stop, take


def liquidity_factor(ctx: BotContext, symbol: str) -> float:
    df = fetch_minute_df_safe(ctx, symbol)
    if df is None or df.empty:
        return 0.0
    if "volume" not in df.columns:
        return 0.0
    avg_vol = df["volume"].tail(30).mean()
    try:
        req = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
        quote: Quote = ctx.data_client.get_stock_latest_quote(req)
        spread = (
            (quote.ask_price - quote.bid_price)
            if quote.ask_price and quote.bid_price
            else 0.0
        )
    except APIError as e:
        logger.warning(f"[liquidity_factor] Alpaca quote failed for {symbol}: {e}")
        spread = 0.0
    except Exception:
        spread = 0.0
    vol_score = min(1.0, avg_vol / ctx.volume_threshold) if avg_vol else 0.0
    spread_score = max(0.0, 1 - spread / 0.05)
    return max(0.0, min(1.0, vol_score * spread_score))


def fractional_kelly_size(
    ctx: BotContext,
    balance: float,
    price: float,
    atr: float,
    win_prob: float,
    payoff_ratio: float = 1.5,
) -> int:
    # Volatility throttle: if SPY ATR > 2σ, cut kelly in half
    if is_high_vol_thr_spy():
        base_frac = ctx.kelly_fraction * 0.5
    else:
        base_frac = ctx.kelly_fraction
    comp = ctx.capital_scaler.compression_factor(balance)
    base_frac *= comp

    if not os.path.exists(PEAK_EQUITY_FILE):
        with portalocker.Lock(PEAK_EQUITY_FILE, "w", timeout=5) as f:
            f.write(str(balance))
        peak_equity = balance
    else:
        with portalocker.Lock(PEAK_EQUITY_FILE, "r+", timeout=5) as f:
            content = f.read().strip()
            peak_equity = float(content) if content else balance
            if balance > peak_equity:
                f.seek(0)
                f.truncate()
                f.write(str(balance))
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
    dollar_cap = ctx.max_position_dollars / price if price > 0 else raw_pos
    size = int(min(raw_pos, cap_pos, risk_cap, dollar_cap, MAX_POSITION_SIZE))
    return max(size, 1)


def vol_target_position_size(
    cash: float, price: float, returns: np.ndarray, target_vol: float = 0.02
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
    """Submit an order using the institutional execution engine."""
    if not market_is_open():
        logger.warning("MARKET_CLOSED_ORDER_SKIP", extra={"symbol": symbol})
        return None
    return exec_engine.execute_order(symbol, qty, side)


def safe_submit_order(api: TradingClient, req) -> Optional[Order]:
    config.reload_env()
    if not market_is_open():
        logger.warning("MARKET_CLOSED_ORDER_SKIP", extra={"symbol": getattr(req, 'symbol', '')})
        return None
    for attempt in range(2):
        try:
            try:
                acct = api.get_account()
            except Exception:
                acct = None
            if acct and getattr(req, "side", "").lower() == "buy":
                price = getattr(req, "limit_price", None)
                if not price:
                    price = getattr(req, "notional", 0)
                need = float(price or 0) * float(getattr(req, "qty", 0))
                if need > float(getattr(acct, "buying_power", 0)):
                    logger.warning(
                        f"insufficient buying power for {req.symbol}: requested {req.qty}, available {acct.buying_power}"
                    )
                    return None
            if getattr(req, "side", "").lower() == "sell":
                try:
                    positions = api.get_all_positions()
                except Exception:
                    positions = []
                avail = next(
                    (float(p.qty) for p in positions if p.symbol == req.symbol), 0.0
                )
                if float(getattr(req, "qty", 0)) > avail:
                    logger.warning(
                        f"insufficient qty available for {req.symbol}: requested {req.qty}, available {avail}"
                    )
                    return None

            order = api.submit_order(order_data=req)
            while getattr(order, "status", None) == OrderStatus.PENDING_NEW:
                time.sleep(0.5)
                order = api.get_order_by_id(order.id)
            logger.info(
                f"Order status for {req.symbol}: {getattr(order, 'status', '')}"
            )
            status = getattr(order, "status", "")
            filled_qty = getattr(order, "filled_qty", "0")
            if status == "filled":
                logger.info(
                    "ORDER_ACK",
                    extra={"symbol": req.symbol, "order_id": getattr(order, "id", "")},
                )
            elif status == "partially_filled":
                logger.warning(
                    f"Order partially filled for {req.symbol}: {filled_qty}/{getattr(req, 'qty', 0)}"
                )
            elif status in ("rejected", "canceled"):
                logger.error(
                    f"Order for {req.symbol} was {status}: {getattr(order, 'reject_reason', '')}"
                )
            else:
                logger.error(
                    f"Order for {req.symbol} status={status}: {getattr(order, 'reject_reason', '')}"
                )
            return order
        except APIError as e:
            err = e.args[0] or {}
            if "insufficient qty" in str(e).lower():
                logger.warning(f"insufficient qty available for {req.symbol}: {e}")
                return None
            if (
                getattr(req, "side", "").lower() == "sell"
                and err.get("code") == 40310000
            ):
                try:
                    pos_qty = float(api.get_position(req.symbol).qty)
                except Exception:
                    pos_qty = 0.0
                if pos_qty > 0:
                    logger.warning(
                        f"Requested {req.qty} but only {pos_qty} available, selling that"
                    )
                    req.qty = pos_qty
                    order = api.submit_order(order_data=req)
                    while getattr(order, "status", None) == OrderStatus.PENDING_NEW:
                        time.sleep(0.5)
                        order = api.get_order_by_id(order.id)
                    logger.info(
                        f"Order status for {req.symbol}: {getattr(order, 'status', '')}"
                    )
                    return order
                else:
                    logger.warning(f"No shares of {req.symbol} to sell, skipping")
                    return None
            time.sleep(1)
            if attempt == 1:
                logger.warning(f"submit_order failed for {req.symbol}: {e}")
                return None
        except Exception as e:
            time.sleep(1)
            if attempt == 1:
                logger.warning(f"submit_order failed for {req.symbol}: {e}")
                return None
    return None


def poll_order_fill_status(ctx: BotContext, order_id: str, timeout: int = 120) -> None:
    """Poll Alpaca for order fill status until it is no longer open."""
    start = pytime.time()
    while pytime.time() - start < timeout:
        try:
            od = ctx.api.get_order_by_id(order_id)
            status = getattr(od, "status", "")
            filled = getattr(od, "filled_qty", "0")
            if status not in {"new", "accepted", "partially_filled"}:
                logger.info(
                    "ORDER_FINAL_STATUS",
                    extra={
                        "order_id": order_id,
                        "status": status,
                        "filled_qty": filled,
                    },
                )
                return
        except Exception as e:
            logger.warning(f"[poll_order_fill_status] failed for {order_id}: {e}")
            return
        pytime.sleep(3)


def send_exit_order(
    ctx: BotContext, symbol: str, exit_qty: int, price: float, reason: str
) -> None:
    logger.info(
        f"EXIT_SIGNAL | symbol={symbol}  reason={reason}  exit_qty={exit_qty}  price={price}"
    )
    try:
        pos = ctx.api.get_open_position(symbol)
        held_qty = int(pos.qty)
    except Exception:
        held_qty = 0

    if held_qty < exit_qty:
        logger.warning(
            f"No shares available to exit for {symbol} (requested {exit_qty}, have {held_qty})"
        )
        return

    if price <= 0.0:
        req = MarketOrderRequest(
            symbol=symbol,
            qty=exit_qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        safe_submit_order(ctx.api, req)
        return

    limit_order = safe_submit_order(
        ctx.api,
        LimitOrderRequest(
            symbol=symbol,
            qty=exit_qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            limit_price=price,
        ),
    )
    pytime.sleep(5)
    try:
        o2 = ctx.api.get_order_by_id(limit_order.id)
        if getattr(o2, "status", "") in {"new", "accepted", "partially_filled"}:
            ctx.api.cancel_order_by_id(limit_order.id)
            safe_submit_order(
                ctx.api,
                MarketOrderRequest(
                    symbol=symbol,
                    qty=exit_qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                ),
            )
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
    ctx: BotContext, symbol: str, total_qty: int, side: str, duration: int = 300
) -> None:
    start_time = pytime.time()
    placed = 0
    while placed < total_qty and pytime.time() - start_time < duration:
        df = fetch_minute_df_safe(ctx, symbol)
        if df is None or df.empty:
            logger.warning(
                "[VWAP] missing bars, aborting VWAP slice", extra={"symbol": symbol}
            )
            break
        vwap_price = ta.vwap(df["high"], df["low"], df["close"], df["volume"]).iloc[-1]
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            quote: Quote = ctx.data_client.get_stock_latest_quote(req)
            spread = (
                (quote.ask_price - quote.bid_price)
                if quote.ask_price and quote.bid_price
                else 0.0
            )
        except APIError as e:
            logger.warning(f"[vwap_slice] Alpaca quote failed for {symbol}: {e}")
            spread = 0.0
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
                order = safe_submit_order(
                    ctx.api,
                    LimitOrderRequest(
                        symbol=symbol,
                        qty=slice_qty,
                        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                        time_in_force=TimeInForce.IOC,
                        limit_price=round(vwap_price, 2),
                    ),
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
                    _slippage_log.append(
                        (symbol, vwap_price, fill_price, datetime.now(timezone.utc))
                    )
                    with slippage_lock:
                        try:
                            with open(SLIPPAGE_LOG_FILE, "a", newline="") as sf:
                                csv.writer(sf).writerow(
                                    [
                                        datetime.now(timezone.utc).isoformat(),
                                        symbol,
                                        vwap_price,
                                        fill_price,
                                        slip,
                                    ]
                                )
                        except Exception as e:
                            logger.warning(f"Failed to append slippage log: {e}")
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
    max_backoff_interval=300,
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
        df = fetch_minute_df_safe(ctx, symbol)
        if df is None or df.empty:
            retries += 1
            if retries > cfg.max_retries:
                logger.warning(
                    f"[pov_submit] no minute data after {cfg.max_retries} retries, aborting",
                    extra={"symbol": symbol},
                )
                return False
            logger.warning(
                f"[pov_submit] missing bars, retry {retries}/{cfg.max_retries} in {interval:.1f}s",
                extra={"symbol": symbol},
            )
            sleep_time = interval * (0.8 + 0.4 * random.random())
            pytime.sleep(sleep_time)
            interval = min(interval * cfg.backoff_factor, cfg.max_backoff_interval)
            continue
        retries = 0
        interval = cfg.sleep_interval

        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            quote: Quote = ctx.data_client.get_stock_latest_quote(req)
            spread = (
                (quote.ask_price - quote.bid_price)
                if quote.ask_price and quote.bid_price
                else 0.0
            )
        except APIError as e:
            logger.warning(f"[pov_submit] Alpaca quote failed for {symbol}: {e}")
            spread = 0.0
        except Exception:
            spread = 0.0

        vol = df["volume"].iloc[-1]
        if spread > 0.05:
            slice_qty = min(int(vol * cfg.pct * 0.5), total_qty - placed)
        else:
            slice_qty = min(int(vol * cfg.pct), total_qty - placed)

        if slice_qty < 1:
            logger.debug(
                f"[pov_submit] slice_qty<1 (vol={vol}), waiting",
                extra={"symbol": symbol},
            )
            pytime.sleep(cfg.sleep_interval * (0.8 + 0.4 * random.random()))
            continue
        try:
            submit_order(ctx, symbol, slice_qty, side)
        except Exception as e:
            logger.exception(
                f"[pov_submit] submit_order failed on slice, aborting: {e}",
                extra={"symbol": symbol},
            )
            return False
        placed += slice_qty
        logger.info(
            "POV_SLICE_PLACED",
            extra={"symbol": symbol, "slice_qty": slice_qty, "placed": placed},
        )
        pytime.sleep(cfg.sleep_interval * (0.8 + 0.4 * random.random()))
    logger.info("POV_SUBMIT_COMPLETE", extra={"symbol": symbol, "placed": placed})
    return True


def maybe_pyramid(
    ctx: BotContext,
    symbol: str,
    entry_price: float,
    current_price: float,
    atr: float,
    prob: float,
):
    """Add to a winning position when probability remains high."""
    profit = (current_price - entry_price) if entry_price else 0
    if profit > 2 * atr and prob >= 0.75:
        try:
            pos = ctx.api.get_open_position(symbol)
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
    ctx: BotContext, symbol: str, price: float, atr: float, win_prob: float
) -> int:
    cash = float(ctx.api.get_account().cash)
    cap_pct = ctx.params.get("CAPITAL_CAP", CAPITAL_CAP)
    cap_sz = int((cash * cap_pct) / price) if price > 0 else 0
    df = ctx.data_fetcher.get_daily_df(ctx, symbol)
    rets = (
        df["close"].pct_change().dropna().values
        if df is not None and not df.empty
        else np.array([0.0])
    )
    kelly_sz = fractional_kelly_size(ctx, cash, price, atr, win_prob)
    vol_sz = vol_target_position_size(cash, price, rets, target_vol=0.02)
    dollar_cap = ctx.max_position_dollars / price if price > 0 else kelly_sz
    base = int(min(kelly_sz, vol_sz, cap_sz, dollar_cap, MAX_POSITION_SIZE))
    factor = max(0.5, min(1.5, 1 + (win_prob - 0.5)))
    liq = liquidity_factor(ctx, symbol)
    if liq < 0.2:
        return 0
    size = int(base * factor * liq)
    return max(size, 1)


def execute_entry(ctx: BotContext, symbol: str, qty: int, side: str) -> None:
    buying_pw = float(ctx.api.get_account().buying_power)
    if buying_pw <= 0:
        logger.info("NO_BUYING_POWER", extra={"symbol": symbol})
        return
    if qty is None or qty <= 0 or not np.isfinite(qty):
        logger.error(
            f"Invalid order quantity for {symbol}: {qty}. Skipping order and logging input data."
        )
        # Optionally, log signal, price, and input features here for debug
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

    raw = fetch_minute_df_safe(ctx, symbol)
    if raw is None or raw.empty:
        logger.warning("NO_MINUTE_BARS_POST_ENTRY", extra={"symbol": symbol})
        return
    df_ind = prepare_indicators(raw, freq="intraday")
    if df_ind.empty:
        logger.warning("INSUFFICIENT_INDICATORS_POST_ENTRY", extra={"symbol": symbol})
        return
    entry_price = get_latest_close(df_ind)
    ctx.trade_logger.log_entry(symbol, entry_price, qty, side, "", "", confidence=0.5)

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
        now_pac,
        mo,
        mc,
        max_factor=tp_factor,
        min_factor=0.5,
    )
    with targets_lock:
        ctx.stop_targets[symbol] = stop
        ctx.take_profit_targets[symbol] = take


def execute_exit(ctx: BotContext, state: BotState, symbol: str, qty: int) -> None:
    if qty is None or not np.isfinite(qty) or qty <= 0:
        logger.error(f"Invalid order size for {symbol}: {qty}")
        return
    raw = fetch_minute_df_safe(ctx, symbol)
    exit_price = get_latest_close(raw) if raw is not None else 1.0
    send_exit_order(ctx, symbol, qty, exit_price, "manual_exit")
    ctx.trade_logger.log_exit(state, symbol, exit_price)
    on_trade_exit_rebalance(ctx)
    with targets_lock:
        ctx.take_profit_targets.pop(symbol, None)
        ctx.stop_targets.pop(symbol, None)


def exit_all_positions(ctx: BotContext) -> None:
    for pos in ctx.api.get_all_positions():
        qty = abs(int(pos.qty))
        if qty:
            send_exit_order(ctx, pos.symbol, qty, 0.0, "eod_exit")
            logger.info("EOD_EXIT", extra={"symbol": pos.symbol, "qty": qty})


# ─── L. SIGNAL & TRADE LOGIC ───────────────────────────────────────────────────
def signal_and_confirm(
    ctx: BotContext, symbol: str, df: pd.DataFrame, model
) -> Tuple[int, float, str]:
    sig, conf, strat = ctx.signal_manager.evaluate(ctx, df, symbol, model)
    if sig == -1 or conf < CONF_THRESHOLD:
        logger.debug(
            "SKIP_LOW_SIGNAL", extra={"symbol": symbol, "sig": sig, "conf": conf}
        )
        return -1, 0.0, ""
    return sig, conf, strat


def pre_trade_checks(
    ctx: BotContext, state: BotState, symbol: str, balance: float, regime_ok: bool
) -> bool:
    # Streak kill-switch check
    if state.streak_halt_until and datetime.now(PACIFIC) < state.streak_halt_until:
        logger.info(
            "SKIP_STREAK_HALT",
            extra={"symbol": symbol, "until": state.streak_halt_until},
        )
        return False
    if check_pdt_rule(ctx):
        logger.info("SKIP_PDT_RULE", extra={"symbol": symbol})
        return False
    if check_halt_flag():
        logger.info("SKIP_HALT_FLAG", extra={"symbol": symbol})
        return False
    if check_daily_loss(ctx, state):
        logger.info("SKIP_DAILY_LOSS", extra={"symbol": symbol})
        return False
    if check_weekly_loss(ctx, state):
        logger.info("SKIP_WEEKLY_LOSS", extra={"symbol": symbol})
        return False
    if too_many_positions(ctx):
        logger.info("SKIP_TOO_MANY_POSITIONS", extra={"symbol": symbol})
        return False
    if too_correlated(ctx, symbol):
        logger.info("SKIP_HIGH_CORRELATION", extra={"symbol": symbol})
        return False
    return ctx.data_fetcher.get_daily_df(ctx, symbol) is not None


def should_enter(
    ctx: BotContext, state: BotState, symbol: str, balance: float, regime_ok: bool
) -> bool:
    return pre_trade_checks(ctx, state, symbol, balance, regime_ok)


def should_exit(
    ctx: BotContext, symbol: str, price: float, atr: float
) -> Tuple[bool, int, str]:
    try:
        pos = ctx.api.get_open_position(symbol)
        current_qty = int(pos.qty)
    except Exception:
        current_qty = 0

    rebalance_time = ctx.rebalance_buys.get(symbol)
    recent_rebalance = False
    if rebalance_time:
        if datetime.now(timezone.utc) - rebalance_time < timedelta(
            seconds=REBALANCE_HOLD_SECONDS
        ):
            recent_rebalance = True
        else:
            ctx.rebalance_buys.pop(symbol, None)

    if recent_rebalance:
        return False, 0, ""

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
    if (action == "exit_long" and current_qty > 0) or (
        action == "exit_short" and current_qty < 0
    ):
        return True, abs(current_qty), "trailing_stop"

    return False, 0, ""


def _safe_trade(
    ctx: BotContext,
    symbol: str,
    balance: float,
    model: RandomForestClassifier,
    regime_ok: bool,
) -> bool:
    try:
        return trade_logic(ctx, symbol, balance, model, regime_ok)
    except RetryError as e:
        logger.warning(
            f"[trade_logic] retries exhausted for {symbol}: {e}",
            extra={"symbol": symbol},
        )
        return False
    except APIError as e:
        msg = str(e).lower()
        if "insufficient buying power" in msg or "potential wash trade" in msg:
            logger.warning(
                f"[trade_logic] skipping {symbol} due to APIError: {e}",
                extra={"symbol": symbol},
            )
            return False
        else:
            logger.exception(f"[trade_logic] APIError for {symbol}: {e}")
            return False
    except Exception:
        logger.exception(f"[trade_logic] unhandled exception for {symbol}")
        return False


def trade_logic(
    ctx: BotContext, symbol: str, balance: float, model, regime_ok: bool
) -> bool:
    """
    Core per-symbol logic: fetch data, compute features, evaluate signals, enter/exit orders.
    """
    logger.info(f"PROCESSING_SYMBOL | symbol={symbol}")

    # Run pre-trade checks that enforce PDT, halt flags, market regime, etc.
    if not pre_trade_checks(ctx, symbol, balance, regime_ok):
        logger.debug("SKIP_PRE_TRADE_CHECKS", extra={"symbol": symbol})
        return False

    try:
        raw_df = fetch_minute_df_safe(ctx, symbol)
    except APIError as e:
        msg = str(e).lower()
        if "subscription does not permit querying recent sip data" in msg:
            logger.debug(f"{symbol}: minute fetch failed, falling back to daily.")
            raw_df = ctx.data_fetcher.get_daily_df(ctx, symbol)
            if raw_df is None or raw_df.empty:
                logger.debug(f"{symbol}: no daily data either; skipping.")
                logger.info(f"SKIP_NO_PRICE_DATA | {symbol}")
                return False
        else:
            raise
    if raw_df is None or raw_df.empty:
        logger.info(f"SKIP_NO_PRICE_DATA | {symbol}")
        return False

    feat_df = prepare_indicators(raw_df, freq="intraday")
    if feat_df.empty:
        logger.debug(f"SKIP_INSUFFICIENT_FEATURES | symbol={symbol}")
        return True

    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    else:
        feature_names = [
            "rsi",
            "macd",
            "atr",
            "vwap",
            "macds",
            "ichimoku_conv",
            "ichimoku_base",
            "stochrsi",
        ]

    missing = [f for f in feature_names if f not in feat_df.columns]
    if missing:
        logger.info(f"SKIP_MISSING_FEATURES | symbol={symbol}  missing={missing}")
        return True

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
    if final_score is None or not np.isfinite(final_score) or final_score == 0:
        logger.error(
            f"Invalid or empty signal for {symbol}. Data: {raw_df.describe() if not raw_df.empty else 'EMPTY'}"
        )
        return True

    try:
        pos = ctx.api.get_open_position(symbol)
        current_qty = int(pos.qty)
    except Exception:
        current_qty = 0

    reb_time = ctx.rebalance_buys.get(symbol)
    recent_rebal = False
    if reb_time:
        if datetime.now(timezone.utc) - reb_time < timedelta(
            seconds=REBALANCE_HOLD_SECONDS
        ):
            recent_rebal = True
        else:
            ctx.rebalance_buys.pop(symbol, None)

    # Exit: bearish reversal for longs
    if (
        not recent_rebal
        and final_score < 0
        and current_qty > 0
        and abs(conf) >= CONF_THRESHOLD
    ):
        price = get_latest_close(feat_df)
        logger.info(
            f"SIGNAL_REVERSAL_EXIT | symbol={symbol}  final_score={final_score:.4f}  confidence={conf:.4f}"
        )
        send_exit_order(ctx, symbol, current_qty, price, "reversal")
        ctx.trade_logger.log_exit(state, symbol, price)
        with targets_lock:
            ctx.stop_targets.pop(symbol, None)
            ctx.take_profit_targets.pop(symbol, None)
        return

    # Exit: bullish reversal for shorts
    if (
        not recent_rebal
        and final_score > 0
        and current_qty < 0
        and abs(conf) >= CONF_THRESHOLD
    ):
        price = get_latest_close(feat_df)
        logger.info(
            f"SIGNAL_BULLISH_EXIT | symbol={symbol}  final_score={final_score:.4f}  confidence={conf:.4f}"
        )
        send_exit_order(ctx, symbol, abs(current_qty), price, "reversal")
        ctx.trade_logger.log_exit(state, symbol, price)
        with targets_lock:
            ctx.stop_targets.pop(symbol, None)
            ctx.take_profit_targets.pop(symbol, None)
        return

    # Entry: bullish
    if final_score > 0 and conf >= BUY_THRESHOLD and current_qty == 0:
        current_price = get_latest_close(feat_df)
        target_weight = ctx.portfolio_weights.get(symbol, 0.0)
        raw_qty = (
            int(balance * target_weight / current_price) if current_price > 0 else 0
        )

        if raw_qty is None or not np.isfinite(raw_qty) or raw_qty <= 0:
            logger.error(
                f"Invalid order size for {symbol}: {raw_qty}. Signal: {final_score}, Price: {current_price}, Data: {raw_df.describe() if not raw_df.empty else 'EMPTY'}"
            )
            return True

        logger.info(
            f"SIGNAL_BUY | symbol={symbol}  final_score={final_score:.4f}  "
            f"confidence={conf:.4f}  qty={raw_qty}"
        )

        if not sector_exposure_ok(ctx, symbol, raw_qty, current_price):
            logger.info("SKIP_SECTOR_CAP", extra={"symbol": symbol})
            return True

        order = submit_order(ctx, symbol, raw_qty, "buy")
        if order is None:
            logger.debug(f"TRADE_LOGIC_NO_ORDER | symbol={symbol}")
        else:
            logger.debug(
                f"TRADE_LOGIC_ORDER_PLACED | symbol={symbol}  order_id={order.id}"
            )
            ctx.trade_logger.log_entry(
                symbol,
                current_price,
                raw_qty,
                "buy",
                strat,
                signal_tags=strat,
                confidence=conf,
            )

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
                now=now_pac,
                market_open=mo,
                market_close=mc,
                max_factor=tp_factor,
                min_factor=0.5,
            )
            with targets_lock:
                ctx.stop_targets[symbol] = stop
                ctx.take_profit_targets[symbol] = take

        return True

    # Entry: bearish short
    if final_score < 0 and conf >= BUY_THRESHOLD and current_qty == 0:
        current_price = get_latest_close(feat_df)
        atr = feat_df["atr"].iloc[-1]
        qty = calculate_entry_size(ctx, symbol, current_price, atr, conf)

        try:
            asset = ctx.api.get_asset(symbol)
            if hasattr(asset, "shortable") and not asset.shortable:
                logger.info(f"SKIP_NOT_SHORTABLE | symbol={symbol}")
                return True
            avail = getattr(asset, "shortable_shares", None)
            if avail is not None:
                qty = min(qty, int(avail))
        except Exception:
            pass

        if qty is None or not np.isfinite(qty) or qty <= 0:
            logger.error(
                f"Invalid order size for {symbol}: {qty}. Signal: {final_score}, Price: {current_price}, Data: {raw_df.describe() if not raw_df.empty else 'EMPTY'}"
            )
            return True

        logger.info(
            f"SIGNAL_SHORT | symbol={symbol}  final_score={final_score:.4f}  "
            f"confidence={conf:.4f}  qty={qty}"
        )
        if not sector_exposure_ok(ctx, symbol, qty, current_price):
            logger.info("SKIP_SECTOR_CAP", extra={"symbol": symbol})
            return True

        order = submit_order(ctx, symbol, qty, "sell")
        if order is None:
            logger.debug(f"TRADE_LOGIC_NO_ORDER | symbol={symbol}")
        else:
            logger.debug(
                f"TRADE_LOGIC_ORDER_PLACED | symbol={symbol}  order_id={order.id}"
            )
            ctx.trade_logger.log_entry(
                symbol,
                current_price,
                qty,
                "sell",
                strat,
                signal_tags=strat,
                confidence=conf,
            )

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
                now=now_pac,
                market_open=mo,
                market_close=mc,
                max_factor=tp_factor,
                min_factor=0.5,
            )
            stop, take = long_take, long_stop
            with targets_lock:
                ctx.stop_targets[symbol] = stop
                ctx.take_profit_targets[symbol] = take

        return True

    # If holding, check for stops/take/trailing
    if current_qty != 0:
        price = get_latest_close(feat_df)
        atr = feat_df["atr"].iloc[-1]
        if not recent_rebal:
            should_exit_flag, exit_qty, reason = should_exit(ctx, symbol, price, atr)
        else:
            should_exit_flag, exit_qty, reason = False, 0, ""
        if should_exit_flag and exit_qty > 0:
            logger.info(
                f"EXIT_SIGNAL | symbol={symbol}  reason={reason}  "
                f"exit_qty={exit_qty}  price={price:.4f}"
            )
            send_exit_order(ctx, symbol, exit_qty, price, reason)
            ctx.trade_logger.log_exit(state, symbol, price)
            try:
                pos_after = ctx.api.get_open_position(symbol)
                if int(pos_after.qty) == 0:
                    with targets_lock:
                        ctx.stop_targets.pop(symbol, None)
                        ctx.take_profit_targets.pop(symbol, None)
            except Exception:
                pass
        else:
            try:
                pos = ctx.api.get_open_position(symbol)
                entry_price = float(pos.avg_entry_price)
                maybe_pyramid(ctx, symbol, entry_price, price, atr, conf)
            except Exception:
                pass
        return True

    # Else hold / no action
    logger.info(
        f"SKIP_LOW_OR_NO_SIGNAL | symbol={symbol}  "
        f"final_score={final_score:.4f}  confidence={conf:.4f}"
    )
    return True


def compute_portfolio_weights(symbols: List[str]) -> Dict[str, float]:
    """Equal-weight portfolio using a dummy-price fallback for missing data."""
    n = len(symbols)
    if n == 0:
        logger.warning("No tickers to weight—skipping.")
        return {}

    prices = [get_latest_close(ctx.data_fetcher.get_daily_df(ctx, s)) for s in symbols]
    inv_prices = [1.0 / p if p > 0 else 1.0 for p in prices]
    total_inv = sum(inv_prices)
    weights = {s: inv / total_inv for s, inv in zip(symbols, inv_prices)}
    logger.info("PORTFOLIO_WEIGHTS", extra={"weights": weights})
    return weights


def on_trade_exit_rebalance(ctx: BotContext) -> None:
    current = compute_portfolio_weights(list(ctx.portfolio_weights.keys()))
    old = ctx.portfolio_weights
    drift = max(abs(current[s] - old.get(s, 0)) for s in current) if current else 0
    if drift <= 0.1:
        return True
    with portfolio_lock:  # FIXED: protect shared portfolio state
        ctx.portfolio_weights = current
    total_value = float(ctx.api.get_account().portfolio_value)
    for sym, w in current.items():
        target_dollar = w * total_value
        raw = fetch_minute_df_safe(ctx, sym)
        price = get_latest_close(raw) if raw is not None else 1.0
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
    if not hasattr(df1, "loc") or "close" not in df1.columns:
        raise ValueError(
            f"pair_trade_signal: df1 for {sym1} is invalid or missing 'close'"
        )
    if not hasattr(df2, "loc") or "close" not in df2.columns:
        raise ValueError(
            f"pair_trade_signal: df2 for {sym2} is invalid or missing 'close'"
        )
    df = pd.concat([df1["close"], df2["close"]], axis=1).dropna()
    if df.empty:
        return ("no_signal", 0)
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
def fetch_data(
    ctx: BotContext, symbols: List[str], period: str, interval: str
) -> Optional[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    now = datetime.now(timezone.utc)
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
                ohlc = finnhub_client.stock_candle(
                    sym, resolution=interval, _from=unix_from, to=unix_to
                )
            except FinnhubAPIException as e:
                logger.warning(f"[fetch_data] {sym} error: {e}")
                continue

            if not ohlc or ohlc.get("s") != "ok":
                continue

            df_sym = pd.DataFrame(
                {
                    "open": ohlc.get("o", []),
                    "high": ohlc.get("h", []),
                    "low": ohlc.get("l", []),
                    "close": ohlc.get("c", []),
                    "volume": ohlc.get("v", []),
                },
                index=pd.to_datetime(ohlc.get("t", []), unit="s"),
            )

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
    if model is None:
        raise RuntimeError("MODEL_LOAD_FAILED: null model in " + path)
    logger.info("MODEL_LOADED", extra={"path": path})
    return model


def online_update(state: BotState, symbol: str, X_new, y_new) -> None:
    y_new = np.clip(y_new, -0.05, 0.05)
    if state.updates_halted:
        return
    with model_lock:
        try:
            model_pipeline.partial_fit(X_new, y_new)
        except Exception as e:
            logger.error(f"Online update failed for {symbol}: {e}")
            return
    pred = model_pipeline.predict(X_new)
    online_error = float(np.mean((pred - y_new) ** 2))
    log_metrics(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "online_update",
            "symbol": symbol,
            "error": online_error,
        }
    )
    state.rolling_losses.append(online_error)
    if len(state.rolling_losses) >= 20 and sum(state.rolling_losses[-20:]) > 0.02:
        state.updates_halted = True
        logger.warning("Halting online updates due to 20-trade rolling loss >2%")


def update_signal_weights() -> None:
    try:
        if not os.path.exists(TRADE_LOG_FILE):
            logger.warning("No trades log found; skipping weight update.")
            return
        df = pd.read_csv(TRADE_LOG_FILE).dropna(
            subset=["entry_price", "exit_price", "signal_tags"]
        )
        df["pnl"] = (df["exit_price"] - df["entry_price"]) * df["side"].apply(
            lambda s: 1 if s == "buy" else -1
        )
        df["confidence"] = df.get("confidence", 0.5)
        df["reward"] = df["pnl"] * df["confidence"]
        recent_cut = pd.to_datetime(df["exit_time"], errors="coerce")
        recent_mask = recent_cut >= (datetime.now(timezone.utc) - timedelta(days=30))
        df_recent = df[recent_mask]

        stats_all: Dict[str, List[float]] = {}
        stats_recent: Dict[str, List[float]] = {}
        for _, row in df.iterrows():
            for tag in row["signal_tags"].split("+"):
                stats_all.setdefault(tag, []).append(row["reward"])
        for _, row in df_recent.iterrows():
            for tag in row["signal_tags"].split("+"):
                stats_recent.setdefault(tag, []).append(row["reward"])

        new_weights = {}
        for tag, pnls in stats_all.items():
            overall_wr = np.mean([1 if p > 0 else 0 for p in pnls]) if pnls else 0.0
            recent_wr = (
                np.mean([1 if p > 0 else 0 for p in stats_recent.get(tag, [])])
                if stats_recent.get(tag)
                else overall_wr
            )
            weight = 0.7 * recent_wr + 0.3 * overall_wr
            if recent_wr < 0.4:
                weight *= 0.5
            new_weights[tag] = round(weight, 3)

        ALPHA = 0.2
        if os.path.exists(SIGNAL_WEIGHTS_FILE):
            old = (
                pd.read_csv(SIGNAL_WEIGHTS_FILE).set_index("signal")["weight"].to_dict()
            )
        else:
            old = {}
        merged = {
            tag: round(ALPHA * w + (1 - ALPHA) * old.get(tag, w), 3)
            for tag, w in new_weights.items()
        }
        out_df = pd.DataFrame.from_dict(
            merged, orient="index", columns=["weight"]
        ).reset_index()
        out_df.columns = ["signal", "weight"]
        out_df.to_csv(SIGNAL_WEIGHTS_FILE, index=False)
        logger.info("SIGNAL_WEIGHTS_UPDATED", extra={"count": len(merged)})
    except Exception as e:
        logger.exception(f"update_signal_weights failed: {e}")


def run_meta_learning_weight_optimizer(
    trade_log_path: str = TRADE_LOG_FILE,
    output_path: str = SIGNAL_WEIGHTS_FILE,
    alpha: float = 1.0,
):
    if not meta_lock.acquire(blocking=False):
        logger.warning("METALEARN_SKIPPED_LOCKED")
        return
    try:
        if not os.path.exists(trade_log_path):
            logger.warning("METALEARN_NO_TRADES")
            return

        df = pd.read_csv(trade_log_path).dropna(
            subset=["entry_price", "exit_price", "signal_tags"]
        )
        if df.empty:
            logger.warning("METALEARN_NO_VALID_ROWS")
            return

        df["pnl"] = (df["exit_price"] - df["entry_price"]) * df["side"].map(
            {"buy": 1, "sell": -1}
        )
        df["confidence"] = df.get("confidence", 0.5)
        df["reward"] = df["pnl"] * df["confidence"]
        df["outcome"] = (df["pnl"] > 0).astype(int)

        tags = sorted(set(tag for row in df["signal_tags"] for tag in row.split("+")))
        X = np.array(
            [[int(tag in row.split("+")) for tag in tags] for row in df["signal_tags"]]
        )
        y = df["outcome"].values

        if len(y) < len(tags):
            logger.warning("METALEARN_TOO_FEW_SAMPLES")
            return

        sample_w = df["reward"].abs() + 1e-3
        model = Ridge(alpha=alpha, fit_intercept=True)
        if X.empty:
            logger.warning("META_MODEL_TRAIN_SKIPPED_EMPTY")
            return
        model.fit(X, y, sample_weight=sample_w)
        atomic_joblib_dump(model, META_MODEL_PATH)
        logger.info("META_MODEL_TRAINED", extra={"samples": len(y)})
        log_metrics(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "meta_model_train",
                "samples": len(y),
                "hyperparams": json.dumps({"alpha": alpha}),
                "seed": SEED,
                "model": "Ridge",
                "git_hash": get_git_hash(),
            }
        )

        weights = {
            tag: round(max(0, min(1, w)), 3) for tag, w in zip(tags, model.coef_)
        }
        out_df = pd.DataFrame(list(weights.items()), columns=["signal", "weight"])
        out_df.to_csv(output_path, index=False)
        logger.info("META_WEIGHTS_UPDATED", extra={"weights": weights})
    finally:
        meta_lock.release()


def run_bayesian_meta_learning_optimizer(
    trade_log_path: str = TRADE_LOG_FILE, output_path: str = SIGNAL_WEIGHTS_FILE
):
    if not meta_lock.acquire(blocking=False):
        logger.warning("METALEARN_SKIPPED_LOCKED")
        return
    try:
        if not os.path.exists(trade_log_path):
            logger.warning("METALEARN_NO_TRADES")
            return

        df = pd.read_csv(trade_log_path).dropna(
            subset=["entry_price", "exit_price", "signal_tags"]
        )
        if df.empty:
            logger.warning("METALEARN_NO_VALID_ROWS")
            return

        df["pnl"] = (df["exit_price"] - df["entry_price"]) * df["side"].map(
            {"buy": 1, "sell": -1}
        )
        df["outcome"] = (df["pnl"] > 0).astype(int)

        tags = sorted(set(tag for row in df["signal_tags"] for tag in row.split("+")))
        X = np.array(
            [[int(tag in row.split("+")) for tag in tags] for row in df["signal_tags"]]
        )
        y = df["outcome"].values

        if len(y) < len(tags):
            logger.warning("METALEARN_TOO_FEW_SAMPLES")
            return

        model = BayesianRidge(fit_intercept=True, normalize=True)
        if X.size == 0:
            logger.warning("BAYES_MODEL_TRAIN_SKIPPED_EMPTY")
            return
        model.fit(X, y)
        atomic_joblib_dump(model, abspath("meta_model_bayes.pkl"))
        logger.info("META_MODEL_BAYESIAN_TRAINED", extra={"samples": len(y)})
        log_metrics(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "meta_model_bayes_train",
                "samples": len(y),
                "seed": SEED,
                "model": "BayesianRidge",
                "git_hash": get_git_hash(),
            }
        )

        weights = {
            tag: round(max(0, min(1, w)), 3) for tag, w in zip(tags, model.coef_)
        }
        out_df = pd.DataFrame(list(weights.items()), columns=["signal", "weight"])
        out_df.to_csv(output_path, index=False)
        logger.info("META_WEIGHTS_UPDATED", extra={"weights": weights})
    finally:
        meta_lock.release()


def load_global_signal_performance(
    min_trades: int = 10, threshold: float = 0.4
) -> Optional[Dict[str, float]]:
    if not os.path.exists(TRADE_LOG_FILE):
        logger.info("METALEARN_NO_HISTORY")
        return None
    df = pd.read_csv(TRADE_LOG_FILE).dropna(
        subset=["exit_price", "entry_price", "signal_tags"]
    )
    df["pnl"] = (df.exit_price - df.entry_price) * df.side.apply(
        lambda s: 1 if s == "buy" else -1
    )
    results: Dict[str, List[float]] = {}
    for _, row in df.iterrows():
        for tag in row.signal_tags.split("+"):
            results.setdefault(tag, []).append(row.pnl)
    win_rates = {
        tag: round(np.mean([1 if p > 0 else 0 for p in pnls]), 3)
        for tag, pnls in results.items()
        if len(pnls) >= min_trades
    }
    filtered = {tag: wr for tag, wr in win_rates.items() if wr >= threshold}
    logger.info(
        "METALEARN_FILTERED_SIGNALS", extra={"signals": list(filtered.keys()) or []}
    )
    return filtered


def prepare_indicators(df: pd.DataFrame, freq: str = "daily") -> pd.DataFrame:
    df = df.copy()
    if df.index.name:
        df = df.reset_index().rename(columns={df.index.name: "date"})
    else:
        df = df.reset_index().rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # ensure rolling simple moving averages are available for ML
    df["sma_50"] = df["close"].astype(float).rolling(window=50).mean()
    df["sma_200"] = df["close"].astype(float).rolling(window=200).mean()

    # ── New advanced indicators ───────────────────────────────────────────
    try:
        kc = ta.kc(df["high"], df["low"], df["close"], length=20)
        df["kc_lower"] = kc.iloc[:, 0]
        df["kc_mid"] = kc.iloc[:, 1]
        df["kc_upper"] = kc.iloc[:, 2]
    except Exception:
        df["kc_lower"] = np.nan
        df["kc_mid"] = np.nan
        df["kc_upper"] = np.nan

    df["atr_band_upper"] = df["close"] + 1.5 * df["atr"]
    df["atr_band_lower"] = df["close"] - 1.5 * df["atr"]
    df["avg_vol_20"] = df["volume"].rolling(20).mean()
    df["dow"] = df.index.dayofweek

    try:
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is None or not isinstance(macd, pd.DataFrame):
            raise ValueError("MACD returned None")
        df["macd"] = macd["MACD_12_26_9"]
        df["macds"] = macd["MACDs_12_26_9"]
    except Exception:
        df["macd"] = np.nan
        df["macds"] = np.nan

    # Additional indicators for richer ML features
    try:
        bb = ta.bbands(df["close"], length=20)
        df["bb_upper"] = bb["BBU_20_2.0"]
        df["bb_lower"] = bb["BBL_20_2.0"]
        df["bb_percent"] = bb["BBP_20_2.0"]
    except Exception:
        df["bb_upper"] = np.nan
        df["bb_lower"] = np.nan
        df["bb_percent"] = np.nan

    try:
        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        df["adx"] = adx["ADX_14"]
        df["dmp"] = adx["DMP_14"]
        df["dmn"] = adx["DMN_14"]
    except Exception:
        df["adx"] = np.nan
        df["dmp"] = np.nan
        df["dmn"] = np.nan

    try:
        df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=20)
    except Exception:
        df["cci"] = np.nan

    # Ensure numeric dtype before computing MFI
    df[["high", "low", "close", "volume"]] = df[
        ["high", "low", "close", "volume"]
    ].astype(float)
    try:
        mfi_vals = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)
        df["mfi"] = mfi_vals.astype(float)
    except Exception as e:
        logger.warning(f"prepare_indicators: failed to compute MFI: {e}")
        df["mfi"] = None

    try:
        df["tema"] = ta.tema(df["close"], length=10)
    except Exception:
        df["tema"] = np.nan

    try:
        df["willr"] = ta.willr(df["high"], df["low"], df["close"], length=14)
    except Exception:
        df["willr"] = np.nan

    try:
        psar = ta.psar(df["high"], df["low"], df["close"])
        df["psar_long"] = psar["PSARl_0.02_0.2"]
        df["psar_short"] = psar["PSARs_0.02_0.2"]
    except Exception:
        df["psar_long"] = np.nan
        df["psar_short"] = np.nan

    try:
        ich = ta.ichimoku(high=df["high"], low=df["low"], close=df["close"])
        conv = ich[0] if isinstance(ich, tuple) else ich.iloc[:, 0]
        base = ich[1] if isinstance(ich, tuple) else ich.iloc[:, 1]
        df["ichimoku_conv"] = conv.iloc[:, 0] if hasattr(conv, "iloc") else conv
        df["ichimoku_base"] = base.iloc[:, 0] if hasattr(base, "iloc") else base
    except Exception:
        df["ichimoku_conv"] = np.nan
        df["ichimoku_base"] = np.nan

    try:
        st = ta.stochrsi(df["close"])
        df["stochrsi"] = st["STOCHRSIk_14_14_3_3"]
    except Exception:
        df["stochrsi"] = np.nan

    # --- Multi-timeframe fusion ---
    try:
        df["ret_5m"] = df["close"].pct_change(5)
        df["ret_1h"] = df["close"].pct_change(60)
        df["ret_d"] = df["close"].pct_change(390)
        df["ret_w"] = df["close"].pct_change(1950)
        df["vol_norm"] = (
            df["volume"].rolling(60).mean() / df["volume"].rolling(5).mean()
        )
        df["5m_vs_1h"] = df["ret_5m"] - df["ret_1h"]
        df["vol_5m"] = df["close"].pct_change().rolling(5).std()
        df["vol_1h"] = df["close"].pct_change().rolling(60).std()
        df["vol_d"] = df["close"].pct_change().rolling(390).std()
        df["vol_w"] = df["close"].pct_change().rolling(1950).std()
        df["vol_ratio"] = df["vol_5m"] / df["vol_1h"]
        df["mom_agg"] = df["ret_5m"] + df["ret_1h"] + df["ret_d"]
        df["lag_close_1"] = df["close"].shift(1)
        df["lag_close_3"] = df["close"].shift(3)
    except Exception:
        df["ret_5m"] = df["ret_1h"] = df["ret_d"] = df["ret_w"] = np.nan
        df["vol_norm"] = df["5m_vs_1h"] = np.nan
        df["vol_5m"] = df["vol_1h"] = df["vol_d"] = df["vol_w"] = np.nan
        df["vol_ratio"] = df["mom_agg"] = df["lag_close_1"] = df["lag_close_3"] = np.nan

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    required = [
        "vwap",
        "rsi",
        "atr",
        "macd",
        "macds",
        "ichimoku_conv",
        "ichimoku_base",
        "stochrsi",
    ]
    if freq == "daily":
        df["sma_50"] = ta.sma(df["close"], length=50)
        df["sma_200"] = ta.sma(df["close"], length=200)
        required += ["sma_50", "sma_200"]
        df.dropna(subset=required, how="any", inplace=True)
    else:  # intraday
        df.dropna(subset=required, how="all", inplace=True)
        df.reset_index(drop=True, inplace=True)

    if os.path.exists(INACTIVE_FEATURES_FILE):
        try:
            with open(INACTIVE_FEATURES_FILE) as f:
                inactive = set(json.load(f))
            df.drop(
                columns=[c for c in inactive if c in df.columns],
                inplace=True,
                errors="ignore",
            )
        except Exception:
            pass

    return df


def _compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    feat["rsi"] = ta.rsi(df["close"], length=14)
    feat["macd"] = ta.macd(df["close"], fast=12, slow=26, signal=9)["MACD_12_26_9"]
    feat["vol"] = df["close"].pct_change().rolling(14).std()
    return feat.dropna()


def detect_regime(df: pd.DataFrame) -> str:
    """Simple SMA-based market regime detection."""
    if df is None or df.empty or "close" not in df:
        return "chop"
    close = df["close"].astype(float)
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    if sma50.iloc[-1] > sma200.iloc[-1]:
        return "bull"
    if sma50.iloc[-1] < sma200.iloc[-1]:
        return "bear"
    return "chop"


# Train or load regime model
if os.path.exists(REGIME_MODEL_PATH):
    try:
        with open(REGIME_MODEL_PATH, "rb") as f:
            regime_model = pickle.load(f)
    except Exception as e:
        logger.warning(f"Failed to load regime model: {e}")
        regime_model = RandomForestClassifier(
            n_estimators=RF_ESTIMATORS, max_depth=RF_MAX_DEPTH
        )
else:
    today_date = date.today()
    start_dt = datetime.combine(
        today_date - timedelta(days=365), dt_time.min, timezone.utc
    )
    end_dt = datetime.combine(today_date, dt_time.max, timezone.utc)
    if isinstance(start_dt, tuple):
        start_dt, _tmp = start_dt
    if isinstance(end_dt, tuple):
        _, end_dt = end_dt
    bars_req = StockBarsRequest(
        symbol_or_symbols=[REGIME_SYMBOLS[0]],
        timeframe=TimeFrame.Day,
        start=start_dt,
        end=end_dt,
        limit=1000,
        feed="iex",
    )
    try:
        bars = ctx.data_client.get_stock_bars(bars_req).df
    except APIError as e:
        logger.warning(
            f"[regime_data] SIP failed {start_dt}-{end_dt}: {e}; retrying with IEX"
        )
        bars_req.feed = "iex"
        bars = ctx.data_client.get_stock_bars(bars_req).df
    # 1) If columns are (symbol, field), select our one symbol
    if isinstance(bars.columns, pd.MultiIndex):
        bars = bars.xs(REGIME_SYMBOLS[0], level=0, axis=1)
    else:
        bars = bars.drop(columns=["symbol"], errors="ignore")

    # 2) Fix the index if it's a MultiIndex of (symbol, timestamp)
    if isinstance(bars.index, pd.MultiIndex):
        bars.index = bars.index.get_level_values(1)
    # 3) Or if each index entry is still a 1-tuple
    elif bars.index.dtype == object and isinstance(bars.index[0], tuple):
        bars.index = [t[0] for t in bars.index]

    # 4) Now safely convert to a timezone-naive DatetimeIndex
    idx = safe_to_datetime(bars.index)
    if idx is None:
        logger.warning("Invalid regime data index; skipping regime model train")
        bars = pd.DataFrame()
    else:
        bars.index = idx
    bars = bars.rename(columns=lambda c: c.lower())
    feats = _compute_regime_features(bars)
    labels = (
        (bars["close"] > bars["close"].rolling(200).mean())
        .loc[feats.index]
        .astype(int)
        .rename("label")
    )
    training = feats.join(labels, how="inner").dropna()
    if len(training) >= 50:
        X = training[["atr", "rsi", "macd", "vol"]]
        y = training["label"]
        regime_model = RandomForestClassifier(
            n_estimators=RF_ESTIMATORS, max_depth=RF_MAX_DEPTH
        )
        if X.empty:
            logger.warning("REGIME_MODEL_TRAIN_SKIPPED_EMPTY")
        else:
            regime_model.fit(X, y)
        try:
            atomic_pickle_dump(regime_model, REGIME_MODEL_PATH)
        except Exception as e:
            logger.warning(f"Failed to save regime model: {e}")
        else:
            logger.info("REGIME_MODEL_TRAINED", extra={"rows": len(training)})
    else:
        logger.error(
            f"Not enough valid rows ({len(training)}) to train regime model; using dummy fallback"
        )
        regime_model = RandomForestClassifier(
            n_estimators=RF_ESTIMATORS, max_depth=RF_MAX_DEPTH
        )


def _market_breadth(ctx: BotContext) -> float:
    syms = load_tickers(TICKERS_FILE)[:20]
    up = 0
    total = 0
    for sym in syms:
        df = ctx.data_fetcher.get_daily_df(ctx, sym)
        if df is None or len(df) < 2:
            continue
        total += 1
        if df["close"].iloc[-1] > df["close"].iloc[-2]:
            up += 1
    return up / total if total else 0.5


def detect_regime_state(ctx: BotContext) -> str:
    df = ctx.data_fetcher.get_daily_df(ctx, REGIME_SYMBOLS[0])
    if df is None or len(df) < 200:
        return "sideways"
    atr14 = ta.atr(df["high"], df["low"], df["close"], length=14).iloc[-1]
    atr50 = ta.atr(df["high"], df["low"], df["close"], length=50).iloc[-1]
    high_vol = atr50 > 0 and atr14 / atr50 > 1.5
    sma50 = df["close"].rolling(50).mean().iloc[-1]
    sma200 = df["close"].rolling(200).mean().iloc[-1]
    trend = sma50 - sma200
    breadth = _market_breadth(ctx)
    if high_vol:
        return "high_volatility"
    if abs(trend) / sma200 < 0.005:
        return "sideways"
    if trend > 0 and breadth > 0.55:
        return "trending"
    if trend < 0 and breadth < 0.45:
        return "mean_reversion"
    return "sideways"


def check_market_regime(state: BotState) -> bool:
    state.current_regime = detect_regime_state(ctx)
    return True


def screen_universe(
    candidates: Sequence[str],
    ctx: BotContext,
    lookback: str = "1mo",
    interval: str = "1d",
    top_n: int = 20,
) -> list[str]:
    atrs: Dict[str, float] = {}
    for sym in candidates:
        df = ctx.data_fetcher.get_daily_df(ctx, sym)
        if df is None or len(df) < ATR_LENGTH:
            continue
        df = df[df["volume"] > 100_000]
        if df.empty:
            continue
        series = ta.atr(df["high"], df["low"], df["close"], length=ATR_LENGTH)
        atr_val = series.iloc[-1] if not series.empty else np.nan
        if not pd.isna(atr_val):
            atrs[sym] = float(atr_val)
    ranked = sorted(atrs.items(), key=lambda kv: kv[1], reverse=True)
    return [sym for sym, _ in ranked[:top_n]]


def screen_candidates() -> list[str]:
    """Load tickers and apply universe screening."""
    candidates = load_tickers(TICKERS_FILE)
    return screen_universe(candidates, ctx)


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
    try:
        if not os.path.exists(TRADE_LOG_FILE):
            logger.info("DAILY_SUMMARY_NO_TRADES")
            return
        df = pd.read_csv(TRADE_LOG_FILE).dropna(subset=["entry_price", "exit_price"])
        df["pnl"] = (df.exit_price - df.entry_price) * df["side"].map(
            {"buy": 1, "sell": -1}
        )
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
                "max_drawdown": max_dd,
            },
        )
    except Exception as e:
        logger.exception(f"daily_summary failed: {e}")


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
        rts = df["close"].pct_change().tail(90).reset_index(drop=True)
        returns_df[sym] = rts
    returns_df = returns_df.dropna(axis=1, how="any")
    if returns_df.shape[1] < 2:
        return
    pca = PCA(n_components=3)
    if returns_df.empty:
        logger.warning("PCA_SKIPPED_EMPTY_RETURNS")
        return
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
    with portfolio_lock:  # FIXED: protect shared portfolio state
        for sym in high_load_syms:
            old = ctx.portfolio_weights.get(sym, 0.0)
            ctx.portfolio_weights[sym] = round(old * 0.8, 4)
        # Re-normalize to sum to 1
        total = sum(ctx.portfolio_weights.values())
        if total > 0:
            for sym in ctx.portfolio_weights:
                ctx.portfolio_weights[sym] = round(
                    ctx.portfolio_weights[sym] / total, 4
                )
    logger.info(
        "PCA_ADJUSTMENT_APPLIED",
        extra={"var_explained": round(var_explained, 3), "adjusted": high_load_syms},
    )


def daily_reset(state: BotState) -> None:
    """Reset daily counters and in-memory slippage logs."""
    try:
        config.reload_env()
        _slippage_log.clear()
        state.loss_streak = 0
        logger.info("DAILY_STATE_RESET")
    except Exception as e:
        logger.exception(f"daily_reset failed: {e}")


def _average_reward(n: int = 20) -> float:
    if not os.path.exists(REWARD_LOG_FILE):
        return 0.0
    df = pd.read_csv(REWARD_LOG_FILE).tail(n)
    if df.empty or "reward" not in df.columns:
        return 0.0
    return float(df["reward"].mean())


def _current_drawdown() -> float:
    try:
        with open(PEAK_EQUITY_FILE) as pf:
            peak = float(pf.read().strip() or 0)
        with open(EQUITY_FILE) as ef:
            eq = float(ef.read().strip() or 0)
    except Exception:
        return 0.0
    if peak <= 0:
        return 0.0
    return max(0.0, (peak - eq) / peak)


def update_bot_mode(state: BotState) -> None:
    try:
        avg_r = _average_reward()
        dd = _current_drawdown()
        regime = state.current_regime
        if dd > 0.05 or avg_r < -0.01:
            new_mode = "conservative"
        elif avg_r > 0.05 and regime == "trending":
            new_mode = "aggressive"
        else:
            new_mode = "balanced"
        if new_mode != state.mode_obj.mode:
            state.mode_obj = BotMode(new_mode)
            params.update(state.mode_obj.get_config())
            ctx.kelly_fraction = params.get("KELLY_FRACTION", 0.6)
            logger.info(
                "MODE_SWITCH",
                extra={
                    "new_mode": new_mode,
                    "avg_reward": avg_r,
                    "drawdown": dd,
                    "regime": regime,
                },
            )
    except Exception as e:
        logger.exception(f"update_bot_mode failed: {e}")


def adaptive_risk_scaling(ctx: BotContext) -> None:
    """Adjust risk parameters based on volatility, rewards and drawdown."""
    try:
        vol = _VOL_STATS.get("mean", 0)
        spy_atr = _VOL_STATS.get("last", 0)
        avg_r = _average_reward(30)
        dd = _current_drawdown()
        try:
            equity = float(ctx.api.get_account().equity)
        except Exception:
            equity = 0.0
        ctx.capital_scaler.update(ctx, equity)
        params["CAPITAL_CAP"] = ctx.params["CAPITAL_CAP"]
        frac = params.get("KELLY_FRACTION", 0.6)
        if spy_atr and vol and spy_atr > vol * 1.5:
            frac *= 0.5
        if avg_r < -0.02:
            frac *= 0.7
        if dd > 0.1:
            frac *= 0.5
        ctx.kelly_fraction = round(max(0.2, min(frac, 1.0)), 2)
        params["CAPITAL_CAP"] = round(
            max(0.02, min(0.1, params.get("CAPITAL_CAP", 0.08) * (1 - dd))), 3
        )
        logger.info(
            "RISK_SCALED",
            extra={
                "kelly_fraction": ctx.kelly_fraction,
                "dd": dd,
                "atr": spy_atr,
                "avg_reward": avg_r,
            },
        )
    except Exception as e:
        logger.exception(f"adaptive_risk_scaling failed: {e}")


def check_disaster_halt() -> None:
    try:
        dd = _current_drawdown()
        if dd >= DISASTER_DD_LIMIT:
            with open(HALT_FLAG_PATH, "w") as f:
                f.write("DISASTER " + datetime.now(timezone.utc).isoformat())
            logger.error("DISASTER_HALT_TRIGGERED", extra={"drawdown": dd})
    except Exception as e:
        logger.exception(f"check_disaster_halt failed: {e}")


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
            logger.warning(
                "Daily retraining requested, but retrain_meta_learner is unavailable."
            )
        else:
            if not meta_lock.acquire(blocking=False):
                logger.warning("METALEARN_SKIPPED_LOCKED")
            else:
                try:
                    symbols = load_tickers(TICKERS_FILE)
                    logger.info(
                        f"RETRAINING START for {today_str} on {len(symbols)} tickers..."
                    )
                    valid_symbols = []
                    for symbol in symbols:
                        df_min = fetch_minute_df_safe(ctx, symbol)
                        if df_min is None or df_min.empty:
                            logger.info(
                                f"{symbol} returned no minute data; skipping symbol."
                            )
                            continue
                        valid_symbols.append(symbol)
                    if not valid_symbols:
                        logger.warning(
                            "No symbols returned valid minute data; skipping retraining entirely."
                        )
                    else:
                        force_train = not os.path.exists(MODEL_PATH)
                        if is_market_open():
                            success = retrain_meta_learner(
                                ctx, valid_symbols, force=force_train
                            )
                        else:
                            logger.info(
                                "[retrain_meta_learner] Outside market hours; skipping"
                            )
                            success = False
                        if success:
                            try:
                                with open(marker, "w") as f:
                                    f.write(today_str)
                            except Exception as e:
                                logger.warning(
                                    f"Failed to write retrain marker file: {e}"
                                )
                        else:
                            logger.warning(
                                "Retraining failed; continuing with existing model."
                            )
                finally:
                    meta_lock.release()

    df_train = ctx.data_fetcher.get_daily_df(ctx, REGIME_SYMBOLS[0])
    if df_train is not None and not df_train.empty:
        X_train = (
            df_train[["open", "high", "low", "close", "volume"]]
            .astype(float)
            .iloc[:-1]
            .values
        )
        y_train = df_train["close"].pct_change().shift(-1).fillna(0).values[:-1]
        with model_lock:
            try:
                if len(X_train) == 0:
                    logger.warning("DAILY_MODEL_TRAIN_SKIPPED_EMPTY")
                else:
                    model_pipeline.fit(X_train, y_train)
                    mse = float(np.mean((model_pipeline.predict(X_train) - y_train) ** 2))
                    logger.info("TRAIN_METRIC", extra={"mse": mse})
            except Exception as e:
                logger.error(f"Daily retrain failed: {e}")

        date_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        os.makedirs("models", exist_ok=True)
        path = f"models/sgd_{date_str}.pkl"
        atomic_joblib_dump(model_pipeline, path)
        logger.info(f"Model checkpoint saved: {path}")

        for f in os.listdir("models"):
            if f.endswith(".pkl"):
                dt = datetime.strptime(f.split("_")[1].split(".")[0], "%Y%m%d").replace(
                    tzinfo=timezone.utc
                )
                if datetime.now(timezone.utc) - dt > timedelta(days=30):
                    os.remove(os.path.join("models", f))

        batch_mse = float(np.mean((model_pipeline.predict(X_train) - y_train) ** 2))
        log_metrics(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "daily_retrain",
                "batch_mse": batch_mse,
                "hyperparams": json.dumps(config.SGD_PARAMS),
                "seed": SEED,
                "model": "SGDRegressor",
                "git_hash": get_git_hash(),
            }
        )
        state.updates_halted = False
        state.rolling_losses.clear()

    return model_pipeline


# ─── M. MAIN LOOP & SCHEDULER ─────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/health")
@app.route("/health_check")
def health() -> str:
    """Simple health endpoint for liveness probes."""
    return "OK", 200


def start_healthcheck() -> None:
    port = int(config.get_env("HEALTHCHECK_PORT", "8080"))
    try:
        app.run(host="0.0.0.0", port=port)
    except OSError as e:
        logger.warning(
            f"Healthcheck port {port} in use: {e}. Skipping health-endpoint."
        )
    except Exception as e:
        logger.exception(f"start_healthcheck failed: {e}")


def run_multi_strategy(ctx: BotContext) -> None:
    """Execute all modular strategies via allocator and risk engine."""
    signals_by_strategy: Dict[str, List[TradeSignal]] = {}
    for strat in ctx.strategies:
        try:
            sigs = strat.generate(ctx)
            signals_by_strategy[strat.name] = sigs
        except Exception as e:
            logger.warning(f"Strategy {strat.name} failed: {e}")
    final = ctx.allocator.allocate(signals_by_strategy)
    acct = ctx.api.get_account()
    cash = float(getattr(acct, "cash", 0))
    for sig in final:
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=[sig.symbol])
            quote: Quote = ctx.data_client.get_stock_latest_quote(req)
            price = float(getattr(quote, "ask_price", 0) or 0)
        except APIError as e:
            logger.warning(f"[run_all_trades] quote failed for {sig.symbol}: {e}")
            continue
        qty = ctx.risk_engine.position_size(sig, cash, price)
        if qty is None or not np.isfinite(qty) or qty <= 0:
            bars = fetch_minute_df_safe(ctx, sig.symbol)
            logger.error(
                f"Invalid order size for {sig.symbol}: {qty}. Signal: {sig.weight}, Price: {price}, Data: {bars.describe() if not bars.empty else 'EMPTY'}"
            )
            continue
        ctx.execution_engine.execute_order(
            sig.symbol, qty, sig.side, asset_class=sig.asset_class
        )
        ctx.risk_engine.register_fill(sig)


def run_all_trades_worker(state: BotState, model) -> None:
    """Run the full trading cycle for all candidate symbols.

    This fetches data, executes strategy logic and places orders. The function
    guards against overlapping runs and ensures the market is open before
    proceeding.
    """
    import uuid
    loop_id = str(uuid.uuid4())
    if state.running:
        logger.warning("RUN_ALL_TRADES_SKIPPED_OVERLAP")
        return
    if not is_market_open():
        logger.info("MARKET_CLOSED_NO_FETCH")
        return  # FIXED: skip work when market closed
    state.running = True
    loop_start = time.monotonic()
    try:
        # On each run, clear open orders and correct any position mismatches
        cancel_all_open_orders(ctx)
        audit_positions(ctx)
        try:
            equity = float(ctx.api.get_account().equity)
        except Exception:
            equity = 0.0
        ctx.capital_scaler.update(ctx, equity)
        params["CAPITAL_CAP"] = ctx.params["CAPITAL_CAP"]
        now_utc = pd.Timestamp.utcnow()
        start_date = (now_utc - timedelta(days=5)).date()
        end_date = now_utc.date()

        # Update SPY vol stats first
        compute_spy_vol_stats(ctx)

        logger.info(
            "RUN_ALL_TRADES_START",
            extra={"timestamp": datetime.now(timezone.utc).isoformat()},
        )

        full_watchlist = load_tickers(TICKERS_FILE)
        symbols = screen_candidates()
        if not symbols:
            logger.warning("No tickers passed screening—using fallback watchlist.")
            symbols = full_watchlist
        logger.info("CANDIDATES_SCREENED", extra={"tickers": symbols})
        ctx.tickers = symbols
        with portfolio_lock:  # FIXED: protect shared portfolio state
            ctx.portfolio_weights = compute_portfolio_weights(symbols)

        if check_halt_flag():
            logger.info("TRADING_HALTED_VIA_FLAG")
            return

        acct = ctx.api.get_account()
        current_cash = float(acct.cash)
        regime_ok = check_market_regime(state)

        processed: list[str] = []

        def process_symbol(symbol: str) -> None:
            """Fetch data and execute trading logic for a single symbol."""
            try:
                logger.info(f"PROCESSING_SYMBOL | symbol={symbol}")
                if not is_market_open():
                    logger.info("MARKET_CLOSED_SKIP_SYMBOL", extra={"symbol": symbol})
                    return
                price_df = fetch_minute_df_safe(ctx, symbol)
                if price_df.empty or "close" not in price_df.columns:
                    logger.info(f"SKIP_NO_PRICE_DATA | {symbol}")
                    return
                processed.append(symbol)
                _safe_trade(ctx, symbol, current_cash, model, regime_ok)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}", exc_info=True)

        with ThreadPoolExecutor(max_workers=5) as executor:
            list(executor.map(process_symbol, symbols))

        if not processed:
            logger.warning(
                "No symbols returned any price data; skipping strategy computation."
            )
            return

        run_multi_strategy(ctx)
        logger.info("RUN_ALL_TRADES_COMPLETE")
        try:
            acct = ctx.api.get_account()
            pnl = float(acct.equity) - float(acct.last_equity)
            logger.info(
                "LOOP_PNL",
                extra={"loop_id": loop_id, "pnl": pnl, "mode": "SHADOW" if SHADOW_MODE else "LIVE"},
            )
        except Exception as e:
            logger.warning(f"Failed P&L retrieval: {e}")
    except Exception as e:
        logger.error(f"Exception in trading loop: {e}", exc_info=True)
    finally:
        # Always reset running flag
        state.running = False
        duration = time.monotonic() - loop_start
        logger.info(
            "HEARTBEAT",
            extra={
                "loop_id": loop_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration": duration,
            },
        )


def schedule_run_all_trades(model):
    """Spawn run_all_trades_worker if market is open."""  # FIXED
    if is_market_open():
        t = threading.Thread(
            target=run_all_trades_worker,
            args=(
                state,
                model,
            ),
            daemon=True,
        )
        t.start()
    else:
        logger.info("Market closed—skipping run_all_trades.")


def schedule_run_all_trades_with_delay(model):
    time.sleep(30)
    schedule_run_all_trades(model)


def initial_rebalance(ctx: BotContext, symbols: List[str]) -> None:
    now_pac = datetime.now(PACIFIC)
    if not in_trading_hours(now_pac):
        logger.info("INITIAL_REBALANCE_MARKET_CLOSED")
        return

    acct = ctx.api.get_account()
    equity = float(acct.equity)
    if equity < PDT_EQUITY_THRESHOLD:
        logger.info("INITIAL_REBALANCE_SKIPPED_PDT", extra={"equity": equity})
        return

    cash = float(acct.cash)
    buying_power = float(getattr(acct, "buying_power", cash))
    n = len(symbols)
    if n == 0 or cash <= 0 or buying_power <= 0:
        logger.info("INITIAL_REBALANCE_NO_SYMBOLS_OR_NO_CASH")
        return

    # Determine current UTC time
    now_utc = datetime.now(timezone.utc)
    # If it’s between 00:00 and 00:15 UTC, daily bars may not be published yet.
    if now_utc.hour == 0 and now_utc.minute < 15:
        logger.info("INITIAL_REBALANCE: Too early—daily bars not live yet.")
    else:
        # Gather all symbols that have a valid, nonzero close
        valid_symbols = []
        valid_prices = {}
        for symbol in symbols:
            df_daily = ctx.data_fetcher.get_daily_df(ctx, symbol)
            price = get_latest_close(df_daily)
            if price <= 0:
                # skip symbols with no real close data
                continue
            valid_symbols.append(symbol)
            valid_prices[symbol] = price

        if not valid_symbols:
            logger.warning(
                "INITIAL_REBALANCE: No valid prices for any symbol—skipping rebalance."
            )
        else:
            # Compute equal weights on valid symbols only
            total_capital = cash
            weight_per = 1.0 / len(valid_symbols)

            positions = {p.symbol: int(p.qty) for p in ctx.api.get_all_positions()}

            for sym in valid_symbols:
                price = valid_prices[sym]
                target_qty = int((total_capital * weight_per) // price)
                current_qty = int(positions.get(sym, 0))

                if current_qty < target_qty:
                    qty_to_buy = target_qty - current_qty
                    if qty_to_buy < 1:
                        continue
                    try:
                        submit_order(ctx, sym, qty_to_buy, "buy")
                        logger.info(f"INITIAL_REBALANCE: Bought {qty_to_buy} {sym}")
                        ctx.rebalance_buys[sym] = datetime.now(timezone.utc)
                    except Exception as e:
                        logger.error(
                            f"INITIAL_REBALANCE: Buy failed for {sym}: {repr(e)}"
                        )
                elif current_qty > target_qty:
                    qty_to_sell = current_qty - target_qty
                    if qty_to_sell < 1:
                        continue
                    try:
                        submit_order(ctx, sym, qty_to_sell, "sell")
                        logger.info(f"INITIAL_REBALANCE: Sold {qty_to_sell} {sym}")
                    except Exception as e:
                        logger.error(
                            f"INITIAL_REBALANCE: Sell failed for {sym}: {repr(e)}"
                        )


def main() -> None:
    config.reload_env()
    def _handle_term(signum, frame):
        logger.info("PROCESS_TERMINATION", extra={"signal": signum})
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)

    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["aggressive", "balanced", "conservative"],
        default=BOT_MODE_ENV or "balanced",
    )
    args = parser.parse_args()
    if args.mode != state.mode_obj.mode:
        state.mode_obj = BotMode(args.mode)
        params.update(state.mode_obj.get_config())

    try:
        logger.info(">>> BOT __main__ ENTERED – starting up")

        # --- Market hours check ---
        import pandas_market_calendars as mcal
        from datetime import datetime, timezone

        now_utc = pd.Timestamp.utcnow()
        schedule = nyse.schedule(start_date=now_utc.date(), end_date=now_utc.date())
        if schedule.empty:
            logger.warning(
                f"No NYSE market schedule for {now_utc.date()}; skipping market open/close check."
            )
            market_open = False
        else:
            market_open = nyse.open_at_time(schedule, now_utc)

        if not market_open:
            logger.info(
                "Market is closed. Sleeping for 60 minutes before rechecking."
            )
            import time
            time.sleep(60 * 60)
            sys.exit(0)

        # Try to start Prometheus metrics server; if port is already in use, log and continue
        try:
            start_http_server(9200)
        except OSError as e:
            logger.warning(
                f"Metrics server port 9200 already in use; skipping start_http_server: {e!r}"
            )

        if RUN_HEALTH:
            Thread(target=start_healthcheck, daemon=True).start()

        # Daily jobs
        schedule.every().day.at("00:30").do(
            lambda: Thread(target=daily_summary, daemon=True).start()
        )
        schedule.every().day.at("00:05").do(
            lambda: Thread(target=daily_reset, args=(state,), daemon=True).start()
        )
        schedule.every().day.at("10:00").do(
            lambda: Thread(
                target=run_meta_learning_weight_optimizer, daemon=True
            ).start()
        )
        schedule.every().day.at("02:00").do(
            lambda: Thread(
                target=run_bayesian_meta_learning_optimizer, daemon=True
            ).start()
        )

        # ⮕ Only now import retrain_meta_learner, to avoid circular import / duplicated metrics
        try:
            from retrain import retrain_meta_learner as _tmp_retrain

            retrain_meta_learner = _tmp_retrain
        except ImportError:
            retrain_meta_learner = None
            logger.warning(
                "retrain.py not found or retrain_meta_learner missing. Daily retraining disabled."
            )

        try:
            model = load_or_retrain_daily(ctx)
        except Exception as e:
            logger.error(f"MODEL_LOAD_FAILED: {e}")
            try:
                model = load_model(MODEL_PATH)
            except Exception as e2:
                logger.fatal("Could not load existing model; aborting", exc_info=e2)
                sys.exit(1)
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
        def gather_minute_data_with_delay():
            try:
                time.sleep(30)
                schedule_run_all_trades(model)
            except Exception as e:
                logger.exception(f"gather_minute_data_with_delay failed: {e}")

        schedule.every().minute.do(gather_minute_data_with_delay)
        schedule.every(1).minutes.do(
            lambda: Thread(
                target=validate_open_orders, args=(ctx,), daemon=True
            ).start()
        )
        schedule.every(6).hours.do(
            lambda: Thread(target=update_signal_weights, daemon=True).start()
        )
        schedule.every(30).minutes.do(
            lambda: Thread(target=update_bot_mode, args=(state,), daemon=True).start()
        )
        schedule.every(30).minutes.do(
            lambda: Thread(
                target=adaptive_risk_scaling, args=(ctx,), daemon=True
            ).start()
        )
        schedule.every(REBALANCE_INTERVAL_MIN).minutes.do(
            lambda: Thread(target=maybe_rebalance, args=(ctx,), daemon=True).start()
        )
        schedule.every().day.at("23:55").do(
            lambda: Thread(target=check_disaster_halt, daemon=True).start()
        )

        # Start listening for trade updates in a background thread
        threading.Thread(
            target=stream.run,
            daemon=True,
        ).start()

        # Scheduler loop
        while True:
            try:
                schedule.run_pending()
            except Exception as e:
                logger.exception(f"Scheduler error: {e}")
                time.sleep(5)
                continue
            pytime.sleep(1)

    except Exception as e:
        logger.exception(f"Fatal error in main: {e}")
        raise


if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as exc:
            logger.critical("UNCAUGHT_EXCEPTION: %s", exc, exc_info=True)
            send_slack_alert(f"Bot crashed: {exc}. Restarting in 5s")
            time.sleep(5)
