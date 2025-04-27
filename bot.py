# ─── STANDARD LIBRARIES ─────────────────────────────────────────────────────
import os
import sys
import time
import csv
import re
import threading
import asyncio
import multiprocessing
from datetime import datetime, date, time as dt_time, timedelta
from zoneinfo import ZoneInfo
from functools import lru_cache
from typing import Optional, Tuple
from dataclasses import dataclass, field

# ─── STRUCTURED LOGGING, RETRIES & RATE LIMITING ────────────────────────────
import structlog
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type
)
from ratelimit import limits, sleep_and_retry

# ─── THIRD-PARTY LIBRARIES ────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from flask import Flask
import schedule
import portalocker
from alpaca_trade_api.rest import REST, APIError
from sklearn.ensemble import RandomForestClassifier
import joblib
from dotenv import load_dotenv
import sentry_sdk
from prometheus_client import start_http_server, Counter, Gauge

# ─── STRUCTLOG CONFIG & “TYPED” EXCEPTIONS ──────────────────────────────────
structlog.configure(logger_factory=structlog.stdlib.LoggerFactory())
logger = structlog.get_logger()

class DataFetchError(Exception):   pass

# ─── A. SENTRY ERROR TRACKING ──────────────────────────────────
load_dotenv()
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    traces_sample_rate=0.1,
    environment=os.getenv("BOT_MODE", "live"),
)

# ─── C. PROMETHEUS METRICS ───────────────────────────────────────────────────
orders_total   = Counter('bot_orders_total',   'Total orders sent')
order_failures = Counter('bot_order_failures', 'Order submission failures')
daily_drawdown = Gauge('bot_daily_drawdown',   'Current daily drawdown fraction')

# ─── BOT CONTEXT ─────────────────────────────────────────────────────────────
@dataclass
class BotContext:
    api: REST
    data_fetcher: DataFetcher
    signal_manager: SignalManager
    trade_logger: TradeLogger
    sem: asyncio.Semaphore
    volume_threshold: int
    entry_start_offset: timedelta
    entry_end_offset: timedelta
    market_open: dt_time
    market_close: dt_time
    regime_lookback: int
    regime_atr_threshold: float
    daily_loss_limit: float
    confirmation_count: dict = field(default_factory=dict)
    trailing_extremes: dict   = field(default_factory=dict)
    take_profit_targets: dict = field(default_factory=dict)

# ─── 0) DATA FETCHER WITH CACHING & VOLUME GUARD ─────────────────────────────
class DataFetcher:
    @lru_cache(maxsize=None)
    def get_daily_df(self, ctx: BotContext, symbol: str) -> Optional[pd.DataFrame]:
        """
        Wrap fetch_data and swallow DataFetchError into a None.
        """
        try:
            return fetch_data(ctx, symbol, period="5d", interval="1d")
        except DataFetchError:
            logger.info(f"[SKIP] No daily data for {symbol}")
            return None

    @lru_cache(maxsize=None)
    def get_minute_df(self, ctx: BotContext, symbol: str) -> Optional[pd.DataFrame]:
        try:
            return fetch_data(ctx, symbol, period="1d", interval="1m")
        except DataFetchError:
            logger.info(f"[SKIP] No minute data for {symbol}")
            return None

# ─── SIGNAL MANAGER ───────────────────────────────────────────────────────────
class SignalManager:
    def __init__(self):
        self.momentum_lookback = 5
        self.mean_rev_lookback = 20
        self.mean_rev_zscore_threshold = 1.5
        self.regime_volatility_threshold = REGIME_ATR_THRESHOLD

    def signal_momentum(self, df):
        """
        Momentum signal: +1 if recent pct change >0, 0 if <0
        """
        try:
            df['momentum'] = df['Close'].pct_change(self.momentum_lookback)
            val = df['momentum'].iloc[-1]
            signal = 1 if val > 0 else 0 if val < 0 else -1
            weight = min(abs(val) * 10, 1.0)
            return signal, weight, 'momentum'
        except Exception:
            logger.exception("Error in signal_momentum")
            return -1, 0.0, 'momentum'

    def signal_mean_reversion(self, df):
        """
        Mean reversion based on rolling z‐score of price
        """
        try:
            ma = df['Close'].rolling(self.mean_rev_lookback).mean()
            sd = df['Close'].rolling(self.mean_rev_lookback).std()
            df['zscore'] = (df['Close'] - ma) / sd
            val = df['zscore'].iloc[-1]
            signal = 0 if val > self.mean_rev_zscore_threshold else 1 if val < -self.mean_rev_zscore_threshold else -1
            weight = min(abs(val) / 3, 1.0)
            return signal, weight, 'mean_reversion'
        except Exception:
            logger.exception("Error in signal_mean_reversion")
            return -1, 0.0, 'mean_reversion'

    def signal_ml(self, df, model):
        """
        ML‐based signal using model.predict_proba
        """
        try:
            feats = ['rsi', 'sma_50', 'sma_200', 'macd', 'macds', 'atr']
            X = df[feats].dropna()
            if X.empty:
                return -1, 0.0, 'ml'
            probs = model.predict_proba(X)
            pred = int(probs[-1].argmax())
            conf = float(probs[-1].max())
            return pred, conf, 'ml'
        except Exception:
            logger.exception("Error in signal_ml")
            return -1, 0.0, 'ml'

    def signal_sentiment(self, ticker):
        """
        Simple sentiment heuristic from headlines
        """
        try:
            val = fetch_sentiment(ticker)
            signal = 1 if val > 0 else 0 if val < 0 else -1
            weight = min(abs(val), 0.2)
            return signal, weight, 'sentiment'
        except Exception:
            logger.exception("Error in signal_sentiment")
            return -1, 0.0, 'sentiment'

    def signal_regime(self):
        """
        Regime filter: SPY ATR low → bullish
        """
        try:
            df = fetch_data("SPY", period=f"{REGIME_LOOKBACK}d", interval="1d")
            if df is None or df.empty:
                return -1, 0.0, 'regime'

            atr_series = ta.atr(df['High'], df['Low'], df['Close'], length=REGIME_LOOKBACK)
            if atr_series is None or atr_series.empty or pd.isna(atr_series.iloc[-1]):
                return -1, 0.0, 'regime'

            atr_val = atr_series.iloc[-1]
            signal  = 1 if atr_val < self.regime_volatility_threshold else 0
            return signal, 0.6, 'regime'

        except Exception:
            logger.exception("Error in signal_regime")
            return -1, 0.0, 'regime'

    def signal_stochrsi(self, df):
        """
        StochRSI signal: oversold/overbought thresholds
        """
        try:
            val = df['stochrsi'].iloc[-1]
            signal = 1 if val < 0.2 else 0 if val > 0.8 else -1
            return signal, 0.3, 'stochrsi'
        except Exception:
            logger.exception("Error in signal_stochrsi")
            return -1, 0.0, 'stochrsi'

    def signal_obv(self, df):
        """
        OBV trend signal
        """
        try:
            obv = ta.obv(df['Close'], df['Volume'])
            slope = np.polyfit(range(5), obv.tail(5), 1)[0]
            signal = 1 if slope > 0 else 0 if slope < 0 else -1
            weight = min(abs(slope) / 1e6, 1.0)
            return signal, weight, 'obv'
        except Exception:
            logger.exception("Error in signal_obv")
            return -1, 0.0, 'obv'

    def signal_vsa(self, df):
        """
        Volume Spread Analysis signal
        """
        try:
            body = abs(df['Close'] - df['Open'])
            vsa = df['Volume'] * body
            score = vsa.iloc[-1]
            avg = vsa.rolling(20).mean().iloc[-1]
            signal = 1 if df['Close'].iloc[-1] > df['Open'].iloc[-1] else 0 if df['Close'].iloc[-1] < df['Open'].iloc[-1] else -1
            weight = min(score / avg, 1.0)
            return signal, weight, 'vsa'
        except Exception:
            logger.exception("Error in signal_vsa")
            return -1, 0.0, 'vsa'

    def load_signal_weights(self):
        """
        Load overriden signal weights from CSV
        """
        if not os.path.exists(SIGNAL_WEIGHTS_FILE):
            return {}
        df = pd.read_csv(SIGNAL_WEIGHTS_FILE)
        return {row['signal']: row['weight'] for _, row in df.iterrows()}

    def evaluate(self, df, ticker, model):
        """
        Combine all signals into a final decision
        """
        signals = []
        allowed = load_global_signal_performance()
        fns = [
            self.signal_momentum,
            self.signal_mean_reversion,
            lambda d: self.signal_ml(d, model),
            lambda d: self.signal_sentiment(ticker),
            lambda d: self.signal_regime(),
            self.signal_stochrsi,
            self.signal_obv,
            self.signal_vsa
        ]
        weights = self.load_signal_weights()

        for fn in fns:
            try:
                s, w, lab = fn(df)
                if allowed and lab not in allowed:
                    continue
                if s in [0, 1]:
                    signals.append((s, weights.get(lab, w), lab))
            except Exception:
                continue

        if not signals:
            return -1, 0.0, 'no_signal'

        score = sum((1 if s == 1 else -1) * w for s, w, _ in signals)
        conf = min(abs(score), 1.0)
        final = 1 if score > 0.5 else 0 if score < -0.5 else -1
        label = '+'.join(lbl for _, _, lbl in signals)
        logger.info(f"[SignalManager] {ticker} | final={final} score={score:.2f} | components: {signals}")
        return final, conf, label

# ─── TRADE LOGGER ─────────────────────────────────────────────────────────────
class TradeLogger:
    def __init__(self, path=TRADE_LOG_FILE):
        self.path = path
        if not os.path.exists(path):
            with portalocker.Lock(path, 'w', timeout=1) as f:
                csv.writer(f).writerow([
                    "symbol","entry_time","entry_price",
                    "exit_time","exit_price","qty","side",
                    "strategy","classification","signal_tags"
                ])

    def log_entry(self, symbol, price, qty, side, strategy, signal_tags=""):
        now = datetime.utcnow().isoformat()
        with portalocker.Lock(self.path, 'a', timeout=1) as f:
            csv.writer(f).writerow([symbol, now, price, "","", qty, side, strategy, "", signal_tags])

    def log_exit(self, symbol, exit_price):
        with portalocker.Lock(self.path, 'r+', timeout=1) as f:
            rows = list(csv.reader(f)); header, data = rows[0], rows[1:]
            for row in data:
                if row[0]==symbol and row[3]=="":
                    entry_t = datetime.fromisoformat(row[1])
                    days = (datetime.utcnow()-entry_t).days
                    cls = "day_trade" if days==0 else "swing_trade" if days<5 else "long_trade"
                    row[3],row[4],row[8] = datetime.utcnow().isoformat(), exit_price, cls
                    break
            f.seek(0); f.truncate()
            w = csv.writer(f); w.writerow(header); w.writerows(data)

trade_logger = TradeLogger()

# ─── WRAPPED I/O CALLS ───────────────────────────────────────────────────────
@sleep_and_retry
@limits(calls=60, period=60)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type((requests.RequestException, DataFetchError))
)
def fetch_data(ctx: BotContext, symbol: str, period: str = "1d", interval: str = "1m") -> pd.DataFrame:
    with ctx.sem:
        df = yf.download(symbol, period=period, interval=interval,
                         auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise DataFetchError(f"No data for {symbol}")
    # flatten MultiIndex, reset_index, ffill/bfill...
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    df.ffill().bfill(inplace=True)
    return df

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
            headers={"User-Agent": "AI Trading Bot"}
        )
        r.raise_for_status()
    soup = BeautifulSoup(r.content, "lxml")
    texts = []
    for a in soup.find_all("a"):
        if "8-K" in a.text:
            tr = a.find_parent("tr")
            if tr:
                tds = tr.find_all("td")
                if len(tds) > 3:
                    texts.append(tds[-1].get_text(strip=True))
    return " ".join(texts)

@sleep_and_retry
@limits(calls=30, period=60)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(requests.RequestException)
)
def fetch_sentiment(ctx: BotContext, ticker: str) -> float:
    with ctx.sem:
        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={ticker}&sortBy=publishedAt&language=en"
            f"&pageSize=5&apiKey={NEWS_API_KEY}"
        )
        resp = requests.get(url)
        resp.raise_for_status()
    articles = resp.json().get("articles", [])
    text = " ".join(a["title"] + " " + a.get("description","") for a in articles).lower()
    score = 0.0
    for word, delta in [("beat", .3), ("strong", .3), ("record", .3)]:
        if word in text: score += delta
    for word, delta in [("miss", -.3), ("weak", -.3), ("recall", -.3)]:
        if word in text: score += delta
    for word, delta in [("upgrade", .2), ("buy rating", .2)]:
        if word in text: score += delta
    for word, delta in [("downgrade", -.2), ("sell rating", -.2)]:
        if word in text: score += delta
    return max(min(score, 0.5), -0.5)

@sleep_and_retry
@limits(calls=200, period=60)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type(APIError)
)

def submit_order(
    ctx: BotContext,
    ticker: str,
    qty: int,
    side: str,
    order_type_override: str = None
) -> bool:
    # 1) Count every attempt
    orders_total.inc()

    order_type = order_type_override or ORDER_TYPE
    try:
        with ctx.sem:
            if order_type == 'limit':
                quote = ctx.api.get_latest_quote(ticker)
                bid, ask = quote.bidprice, quote.askprice
                spread = (ask - bid) if ask and bid else 0
                limit_price = (bid + 0.25 * spread) if side == 'buy' else (ask - 0.25 * spread)
                ctx.api.submit_order(
                    symbol=ticker,
                    qty=qty,
                    side=side,
                    type='limit',
                    limit_price=round(limit_price, 2),
                    time_in_force='gtc'
                )
                expected = round(limit_price, 2)
            else:
                quote = ctx.api.get_latest_quote(ticker)
                expected = quote.askprice if side == 'buy' else quote.bidprice
                ctx.api.submit_order(
                    symbol=ticker,
                    qty=qty,
                    side=side,
                    type='market',
                    time_in_force='gtc'
                )

        logger.info(f"[SLIPPAGE] {ticker} expected={expected} side={side} qty={qty}")
        return True

    except Exception:
        # 2) Count any failure
        order_failures.inc()
        logger.exception(f"[submit_order] unexpected error for {ticker}")
        raise  # re-raise so Sentry captures it

# ─── REFACTORED SUB-FUNCTIONS ─────────────────────────────────────────────────
def pre_trade_checks(ctx: BotContext, symbol: str, balance: float) -> bool:
    # 1) HALT / HOURS / DAILY-LOSS
    if check_halt_flag(): 
        logger.info(f"[SKIP] HALT_FLAG – {symbol}")
        return False

    if not within_market_hours(): 
        logger.info(f"[SKIP] Market closed – {symbol}")
        return False

    if check_daily_loss():
        logger.info(f"[SKIP] Daily-loss limit – {symbol}")
        return False

    # 2) REGIME / POSITIONS / CORRELATION
    if not check_market_regime():
        logger.info(f"[SKIP] Market regime – {symbol}")
        return False

    if too_many_positions():
        logger.info(f"[SKIP] Max positions – {symbol}")
        return False

    if too_correlated(symbol):
        logger.info(f"[SKIP] Correlation – {symbol}")
        return False

    # 3) VOLUME GUARD (5d)
    return ctx.data_fetcher.get_daily_df(ctx, symbol) is not None

def is_within_entry_window(ctx: BotContext) -> bool:
    now = now_pacific()
    start = (datetime.combine(now.date(), ctx.market_open) + ctx.entry_start_offset).time()
    end   = (datetime.combine(now.date(), ctx.market_close) - ctx.entry_end_offset).time()

    if not (start <= now.time() <= end):
        logger.info(f"[SKIP] entry window ({start}–{end}), now={now.time()}")
        return False

    return True

def signal_and_confirm(ctx: BotContext, symbol: str, df: pd.DataFrame, model) -> Tuple[int,float,str]:
    sig, conf, strat = ctx.signal_manager.evaluate(df, symbol, model)
    if sig == -1 or conf < CONF_THRESHOLD:
        logger.info(f"[SKIP] {symbol} no/low signal (sig={sig},conf={conf:.2f})")
        return -1, 0.0, ""
    return sig, conf, strat

# ─── 1) ENTRY CHECK ───────────────────────────────────────────────────────────
def should_enter(ctx: BotContext, symbol: str, balance: float) -> bool:
    """All the pre‐trade gates: halts, hours, regime, correlation, volume…"""
    return (
        pre_trade_checks(ctx, symbol, balance)
        and is_within_entry_window(ctx)
    )

# ─── 2) EXIT DECIDER ─────────────────────────────────────────────────────────
def should_exit(ctx: BotContext, symbol: str, price: float) -> Tuple[bool, int, str]:
    """
    Returns (exit?, qty, reason).  
    Checks first for take‐profit scaling, then trailing‐stop.
    """
    try:
        pos = int(ctx.api.get_position(symbol).qty)
    except Exception:
        pos = 0

    # 2a) take-profit
    tp = ctx.take_profit_targets.get(symbol)
    if pos and tp and ((pos > 0 and price >= tp) or (pos < 0 and price <= tp)):
        qty = int(abs(pos) * SCALING_FACTOR)
        return True, qty, "take_profit"

    # 2b) trailing stop
    action = update_trailing_stop(symbol, price, pos)
    if action == "exit_long" and pos > 0:
        return True, pos, "trailing_stop"
    if action == "exit_short" and pos < 0:
        return True, abs(pos), "trailing_stop"

    return False, 0, ""

def execute_exit(ctx: BotContext, symbol: str, qty: int) -> None:
    """Fire market exit and log it."""
    submit_order(ctx, symbol, qty, "sell" if qty>0 else "buy")
    ctx.trade_logger.log_exit(symbol, ctx.data_fetcher.get_minute_df(ctx, symbol)["Close"].iloc[-1])
    # clear profit target so we don't re-exit
    ctx.take_profit_targets.pop(symbol, None)

# ─── 3) ENTRY SIZER & EXECUTOR ────────────────────────────────────────────────
def calculate_entry_size(
    ctx: BotContext,
    symbol: str,
    price: float,
    atr: float,
    win_prob: float,
) -> int:
    """Wrap your Kelly sizing plus any portfolio caps."""
    return fractional_kelly_size(
        balance=float(ctx.api.get_account().cash),
        price=price,
        atr=atr,
        win_prob=win_prob,
    )

def execute_entry(ctx: BotContext, symbol: str, qty: int, side: str) -> None:
    """Submit the order, log entry, and set up a take‐profit target."""
    submit_order(ctx, symbol, qty, side)
    ctx.trade_logger.log_entry(symbol, 
                               price=ctx.data_fetcher.get_minute_df(ctx, symbol)["Close"].iloc[-1],
                               qty=qty,
                               side=side,
                               strategy="",  # you can pass strat here
                               signal_tags="")
    # schedule the take-profit at a fixed factor
    price = ctx.data_fetcher.get_minute_df(ctx, symbol)["Close"].iloc[-1]
    ctx.take_profit_targets[symbol] = price + (TAKE_PROFIT_FACTOR * atr * (1 if side=="buy" else -1))

# ─── NEW, COMPOSED trade_logic ───────────────────────────────────────────────
def trade_logic(ctx: BotContext, symbol: str, balance: float, model) -> None:
    # 1) Entry gating
    if not should_enter(ctx, symbol, balance):
        return

    # 2) Fetch & signal
    df = prepare_indicators(ctx.data_fetcher.get_minute_df(ctx, symbol) or pd.DataFrame())
    sig, conf, strat = signal_and_confirm(ctx, symbol, df, model)
    if sig == -1:
        return

    price, atr = df["Close"].iloc[-1], df["atr"].iloc[-1]

    # 3) Exit
    do_exit, qty, reason = should_exit(ctx, symbol, price)
    if do_exit:
        execute_exit(ctx, symbol, qty)
        return

    # 4) Entry
    size = calculate_entry_size(ctx, symbol, price, atr, conf)
    if size > 0:
        side = "buy" if sig == 1 else "sell"
        # add current pos so we flatten+flip
        try:
            pos = int(ctx.api.get_position(symbol).qty)
        except Exception:
            pos = 0
        execute_entry(ctx, symbol, size + abs(pos), side)

# ─── STRATEGY MODE CONFIGURATION ─────────────────────────────────────────────
class BotMode:
    def __init__(self, mode="balanced"):
        self.mode = mode.lower()
        self.params = self.set_parameters()

    def set_parameters(self):
        if self.mode == "conservative":
            return {"KELLY_FRACTION": 0.3, "CONF_THRESHOLD": 0.8, "CONFIRMATION_COUNT": 3,
                    "TAKE_PROFIT_FACTOR": 1.2, "DAILY_LOSS_LIMIT": 0.05,
                    "CAPITAL_CAP": 0.05, "TRAILING_FACTOR": 1.5}
        elif self.mode == "aggressive":
            return {"KELLY_FRACTION": 0.75, "CONF_THRESHOLD": 0.6, "CONFIRMATION_COUNT": 1,
                    "TAKE_PROFIT_FACTOR": 2.2, "DAILY_LOSS_LIMIT": 0.1,
                    "CAPITAL_CAP": 0.1, "TRAILING_FACTOR": 2.0}
        else:  # balanced
            return {"KELLY_FRACTION": 0.6, "CONF_THRESHOLD": 0.65, "CONFIRMATION_COUNT": 2,
                    "TAKE_PROFIT_FACTOR": 1.8, "DAILY_LOSS_LIMIT": 0.07,
                    "CAPITAL_CAP": 0.08, "TRAILING_FACTOR": 1.8}
    def get_config(self):
        """Return the parameters for the current mode."""
        return self.params

BOT_MODE = os.getenv("BOT_MODE", "balanced")
mode = BotMode(BOT_MODE)
params = mode.get_config()

# ─── LOAD ENV VARS & ASSERT KEYS ─────────────────────────────────────────────
dotenv_path = os.getenv("DOTENV_PATH", ".env")
load_dotenv(dotenv_path=dotenv_path)

ALPACA_API_KEY    = os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
ALPACA_BASE_URL   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    logger.error("❌ Missing Alpaca API credentials; please check .env")
    sys.exit(1)

# ─── TIMEZONE & MARKET-HOURS HELPERS ──────────────────────────────────────────
PACIFIC = ZoneInfo("America/Los_Angeles")

def now_pacific() -> datetime:
    """Get current time in US/Pacific"""
    return datetime.now(PACIFIC)

def within_market_hours() -> bool:
    """
    Return True if current time is within the entry window,
    using US/Pacific.
    """
    now = now_pacific()
    start = datetime.combine(now.date(), MARKET_OPEN, PACIFIC) + ENTRY_START_OFFSET
    end   = datetime.combine(now.date(), MARKET_CLOSE, PACIFIC) - ENTRY_END_OFFSET
    return start <= now <= end

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
MODEL_FILE          = abspath(os.getenv("MODEL_PATH", "trained_model.pkl"))

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
NEWS_API_KEY             = os.getenv("NEWS_API_KEY")
TRAILING_FACTOR          = params["TRAILING_FACTOR"]
TAKE_PROFIT_FACTOR       = params["TAKE_PROFIT_FACTOR"]
SCALING_FACTOR           = 0.5
SECONDARY_TRAIL_FACTOR   = 1.0
ORDER_TYPE               = 'market'
LIMIT_ORDER_SLIPPAGE     = float(os.getenv("LIMIT_ORDER_SLIPPAGE", 0.005))
MAX_POSITION_SIZE        = 1000
DAILY_LOSS_LIMIT         = params["DAILY_LOSS_LIMIT"]
MAX_PORTFOLIO_POSITIONS  = int(os.getenv("MAX_PORTFOLIO_POSITIONS", 15))
CORRELATION_THRESHOLD    = 0.8
MARKET_OPEN              = dt_time(0, 0)
MARKET_CLOSE             = dt_time(23, 59)
VOLUME_THRESHOLD         = 10_000
ENTRY_START_OFFSET       = timedelta(minutes=15)
ENTRY_END_OFFSET         = timedelta(minutes=30)
REGIME_LOOKBACK          = 14
REGIME_ATR_THRESHOLD     = 20.0
MODEL_PATH               = MODEL_FILE
RF_ESTIMATORS            = 225
RF_MAX_DEPTH             = 5
ATR_LENGTH               = 12
CONF_THRESHOLD           = params["CONF_THRESHOLD"]
CONFIRMATION_COUNT       = params["CONFIRMATION_COUNT"]
KELLY_FRACTION           = params["KELLY_FRACTION"]
CAPITAL_CAP              = params["CAPITAL_CAP"]

# ─── GLOBAL STATE ─────────────────────────────────────────────────────────────
api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)

data_fetcher   = DataFetcher()
signal_manager = SignalManager()
trade_logger   = TradeLogger()
semaphore      = asyncio.Semaphore(4)

ctx = BotContext(
    api=api,
    data_fetcher=DataFetcher(),
    signal_manager=signal_manager,
    trade_logger=trade_logger,
    sem=asyncio.Semaphore(4),
    volume_threshold=VOLUME_THRESHOLD,
    entry_start_offset=ENTRY_START_OFFSET,
    entry_end_offset=ENTRY_END_OFFSET,
    market_open=MARKET_OPEN,
    market_close=MARKET_CLOSE,
    regime_lookback=REGIME_LOOKBACK,
    regime_atr_threshold=REGIME_ATR_THRESHOLD,
    daily_loss_limit=DAILY_LOSS_LIMIT,
)

    # 1) Start Prometheus metrics server on port 8000
    start_http_server(8000)

    # 2) Start HTTP healthcheck endpoint in background
    threading.Thread(target=start_healthcheck, daemon=True).start()
    logger.info("Healthcheck endpoint running on port 8080")

    # 3) Schedule end-of-day summary (fill in the function body below)
    def daily_summary():
        # TODO: load trades.csv → compute PnL, drawdown, win rate → post to Slack/email
        pass

    schedule.every().day.at("17:30").do(daily_summary)

# ─── HEALTHCHECK APP ──────────────────────────────────────────────────────────
app = Flask(__name__)
@app.route("/health")
def health():
    return "OK", 200

def start_healthcheck():
    try:
        app.run(host="0.0.0.0", port=8080)
    except Exception:
        logger.exception("Healthcheck server failed")

# ─── META-LEARNING SIGNAL DROPPER ────────────────────────────────────────────
def load_global_signal_performance(min_trades=10, threshold=0.4):
    if not os.path.exists(TRADE_LOG_FILE):
        logger.info("[MetaLearn] no history - allowing all signals")
        return None
    df = pd.read_csv(TRADE_LOG_FILE).dropna(subset=["exit_price","entry_price","signal_tags"])
    df["pnl"] = (df.exit_price - df.entry_price) * df.side.apply(lambda s: 1 if s=="buy" else -1)
    results = {}
    for _, row in df.iterrows():
        for tag in row.signal_tags.split("+"):
            results.setdefault(tag, []).append(row.pnl)
    win_rates = { tag: round(np.mean([1 if p>0 else 0 for p in pnls]),3)
                  for tag,pnls in results.items() if len(pnls)>=min_trades }
    filtered = {tag:wr for tag,wr in win_rates.items() if wr>=threshold}
    logger.info(f"[MetaLearn] Keeping signals: {list(filtered.keys()) or 'none'}")
    return filtered

# ─── UTILITY FUNCTIONS ────────────────────────────────────────────────────────
def load_tickers(path: str = TICKERS_FILE) -> list[str]:
    """Load and return a deduplicated, uppercase list of tickers from CSV."""
    tickers = []
    try:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                t = row[0].strip().upper()
                if t and t not in tickers:
                    tickers.append(t)
    except Exception as e:
        logger.exception(f"[load_tickers] Failed to read {path}: {e}")
    return tickers

def check_daily_loss() -> bool:
    """Return True if cash drawdown from start of day exceeds limit."""
    global daily_loss, day_start_equity
    try:
        cash = float(api.get_account().cash)
    except Exception:
        logger.warning("[check_daily_loss] Could not fetch account cash")
        return False

    today = date.today()
    if day_start_equity is None or day_start_equity[0] != today:
        day_start_equity = (today, cash)
        daily_loss = 0.0
        return False

    loss = (day_start_equity[1] - cash) / day_start_equity[1]
    return loss >= DAILY_LOSS_LIMIT

def too_many_positions() -> bool:
    """Return True if current open positions ≥ portfolio limit."""
    try:
        return len(api.list_positions()) >= MAX_PORTFOLIO_POSITIONS
    except Exception:
        logger.warning("[too_many_positions] Could not fetch positions")
        return False

def check_halt_flag() -> bool:
    """Return True only if halt.flag exists and is <1h old."""
    if not os.path.exists(HALT_FLAG_PATH):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(HALT_FLAG_PATH))
    return age < timedelta(hours=1)

@retry(times=3, delay=0.5)
def check_market_regime() -> bool:
    df = fetch_data("SPY", period="1mo", interval="1d")
    if df is None or df.empty:
        logger.warning("[check_market_regime] No SPY data – failing regime check")
        return False

    logger.debug(f"[check_market_regime] SPY dataframe columns: {df.columns.tolist()}")

    for col in ["High", "Low", "Close"]:
        if col not in df.columns:
            logger.warning(f"[check_market_regime] Missing column '{col}' in SPY data")
            return False

    df.dropna(subset=["High", "Low", "Close"], inplace=True)
    if len(df) < REGIME_LOOKBACK:
        logger.warning(
            f"[check_market_regime] Not enough SPY rows after cleaning: "
            f"{len(df)} – need {REGIME_LOOKBACK}"
        )
        return False

    try:
        atr_series = ta.atr(high=df["High"], low=df["Low"],
                            close=df["Close"], length=REGIME_LOOKBACK)
        atr_val = atr_series.iloc[-1]
        if pd.isna(atr_val):
            logger.warning("[check_market_regime] ATR value is NaN – failing regime check")
            return False
    except Exception as e:
        logger.warning(f"[check_market_regime] ATR calc failed: {e}")
        return False

    vol = df["Close"].pct_change().std()
    logger.debug(f"[check_market_regime] SPY data shape: {df.shape}, "
                 f"ATR: {atr_val:.4f}, Volatility: {vol:.4f}")

    return (atr_val <= REGIME_ATR_THRESHOLD) or (vol <= 0.015)
    
def too_correlated(sym: str) -> bool:
    """
    Computes average pairwise correlation among open positions + this symbol.
    Returns True if above CORRELATION_THRESHOLD.
    """
    if not os.path.exists(TRADE_LOG_FILE):
        return False

    df = pd.read_csv(TRADE_LOG_FILE)
    open_syms = df.loc[df.exit_time == "", "symbol"].unique().tolist() + [sym]

    # Fetch returns
    rets = {}
    for s in open_syms:
        d = fetch_data(s, period="3mo", interval="1d")
        rets[s] = d["Close"].pct_change().dropna()

    # Align lengths
    min_len = min(len(r) for r in rets.values())
    mat = pd.DataFrame({s: rets[s].tail(min_len) for s in open_syms})
    corr_matrix = mat.corr().abs()
    # exclude diagonal
    avg_corr = corr_matrix.where(~np.eye(len(open_syms), dtype=bool)).stack().mean()
    return avg_corr > CORRELATION_THRESHOLD

# ─── ORDER EXECUTION & POSITION SIZING ────────────────────────────────────────
def fractional_kelly_size(balance, price, atr, win_prob,
                           payoff_ratio: float = 1.5,
                           base_frac: float = KELLY_FRACTION) -> int:
    """
    Calculate position size via a fractional Kelly, scaling back on drawdown
    and capping by CAPITAL_CAP and MAX_POSITION_SIZE.
    """
    # Ensure peak‐equity file exists & is locked for concurrency
    if not os.path.exists(PEAK_EQUITY_FILE):
        with portalocker.Lock(PEAK_EQUITY_FILE, 'w', timeout=1) as f:
            f.write(str(balance))
        peak_equity = balance
    else:
        with portalocker.Lock(PEAK_EQUITY_FILE, 'r+', timeout=1) as f:
            content = f.read().strip()
            peak_equity = float(content) if content else balance
            if balance > peak_equity:
                f.seek(0); f.truncate()
                f.write(str(balance))
                peak_equity = balance

    drawdown = (peak_equity - balance) / peak_equity
    # Adjust Kelly fraction based on drawdown
    if drawdown > 0.10:
        frac = 0.3
    elif drawdown > 0.05:
        frac = 0.45
    else:
        frac = base_frac

    # Standard Kelly edge
    edge = win_prob - (1 - win_prob) / payoff_ratio
    kelly = max(edge / payoff_ratio, 0) * frac
    dr = kelly * balance

    # Avoid division by zero
    if atr <= 0:
        return 1

    raw_pos = dr / atr
    cap_pos = (balance * CAPITAL_CAP) / price
    size = int(min(raw_pos, cap_pos, MAX_POSITION_SIZE))
    return max(size, 1)

@sleep_and_retry
@limits(calls=200, period=60)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type(APIError)
)
def submit_order(
    ctx: BotContext,
    ticker: str,
    qty: int,
    side: str,
    order_type_override: str = None
) -> bool:
    order_type = order_type_override or ORDER_TYPE
    try:
        with ctx.sem:
            if order_type == "limit":
                quote = ctx.api.get_latest_quote(ticker)
                bid, ask = quote.bidprice, quote.askprice
                spread = (ask - bid) if (ask and bid) else 0
                limit_price = (bid + 0.25 * spread) if side == "buy" else (ask - 0.25 * spread)
                ctx.api.submit_order(
                    symbol=ticker,
                    qty=qty,
                    side=side,
                    type="limit",
                    limit_price=round(limit_price, 2),
                    time_in_force="gtc",
                )
                expected = round(limit_price, 2)
            else:
                quote = ctx.api.get_latest_quote(ticker)
                expected = quote.askprice if side == "buy" else quote.bidprice
                ctx.api.submit_order(
                    symbol=ticker,
                    qty=qty,
                    side=side,
                    type="market",
                    time_in_force="gtc",
                )

        logger.info(f"[SLIPPAGE] {ticker} expected={expected} side={side} qty={qty}")
        return True

    except APIError as e:
        # handle “requested vs available” case
        m = re.search(r"requested: (\d+), available: (\d+)", str(e))
        if m and int(m.group(2)) > 0:
            available = int(m.group(2))
            ctx.api.submit_order(
                symbol=ticker,
                qty=available,
                side=side,
                type=order_type,
                time_in_force="gtc",
            )
            return True

        logger.warning(f"[submit_order] APIError: {e}")
        raise  # let Tenacity catch & retry

    except Exception
        order_failures.inc()
        logger.exception(f"[submit_order] unexpected error for {ticker}")
        raise

def update_trailing_stop(ticker: str, price: float, qty: int,
                         atr: float, factor: float = TRAILING_FACTOR) -> str:
    """
    Maintain a running “extreme” price and signal exit when price retracts
    by factor * ATR from that extreme.
    Returns one of: "exit_long", "exit_short", or "hold".
    """
    if qty > 0:
        trailing_extremes[ticker] = max(trailing_extremes.get(ticker, price), price)
        if price < trailing_extremes[ticker] - factor * atr:
            return "exit_long"

    elif qty < 0:
        trailing_extremes[ticker] = min(trailing_extremes.get(ticker, price), price)
        if price > trailing_extremes[ticker] + factor * atr:
            return "exit_short"

    return "hold"

# ─── 5) INDICATOR PREPARATION ─────────────────────────────────────────────────
def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["vwap"]    = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"])
    df["sma_50"]  = ta.sma(df["Close"], length=50)
    df["sma_200"] = ta.sma(df["Close"], length=200)
    df["rsi"]     = ta.rsi(df["Close"], length=14)
    macd          = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    df["macd"], df["macds"] = macd["MACD_12_26_9"], macd["MACDs_12_26_9"]
    df["atr"]     = ta.atr(df["High"], df["Low"], df["Close"], length=ATR_LENGTH)

    ich           = ta.ichimoku(df["High"], df["Low"], df["Close"])
    df["ichimoku_base"], df["ichimoku_conv"] = ich["ISA_9"], ich["ISB_26"]
    df["stochrsi"] = ta.stochrsi(df["Close"])["STOCHRSIk_14_14_3_3"]

    df.ffill().bfill(inplace=True)
    df.dropna(subset=[
        "vwap","sma_50","sma_200","rsi",
        "macd","macds","atr",
        "ichimoku_base","ichimoku_conv","stochrsi"
    ], inplace=True)
    return df

# ─── RUNNER & SCHEDULER ──────────────────────────────────────────────────────
def run_all_trades(model):
    """
    Kick off one pass over all tickers in parallel.
    """
    if check_halt_flag():
        logger.info("Trading halted via HALT_FLAG_FILE.")
        return

    try:
        acct = api.get_account()
        current_cash = float(acct.cash)
    except Exception:
        logger.error("Failed to retrieve account balance.")
        return

    # Load last known equity and detect fresh inflows
    if os.path.exists(EQUITY_FILE):
        with open(EQUITY_FILE, "r") as f:
            last_cash = float(f.read().strip())
    else:
        last_cash = current_cash

    global KELLY_FRACTION
    if current_cash > last_cash * 1.03:
        logger.info(f"[REWEIGHT] Cash inflow detected – bump Kelly from {KELLY_FRACTION:.2f} to 0.7")
        KELLY_FRACTION = 0.7
    else:
        KELLY_FRACTION = params["KELLY_FRACTION"]

    # Persist current_cash for next run
    with open(EQUITY_FILE, "w") as f:
        f.write(str(current_cash))

    # Parallelize over tickers
    tickers = load_tickers(TICKERS_FILE)
    if not tickers:
        logger.error("❌ No tickers loaded; please check tickers.csv")
        return

    pool_size = min(len(tickers), 4)
    with multiprocessing.Pool(pool_size) as pool:
        for sym in tickers:
            pool.apply_async(trade_logic, (ctx, sym, current_cash, model))
        pool.close()
        pool.join()

# ─── LOAD MODEL ──────────────────────────────────────────────────────────────
def load_model(path: str = MODEL_PATH):
    """
    Load a pretrained model from disk, or train+persist a fallback RandomForest.
    """
    if os.path.exists(path):
        logger.info(f"Loading trained model from {path}")
        return joblib.load(path)

    logger.info("No model found; training fallback RandomForestClassifier")
    model = RandomForestClassifier(n_estimators=RF_ESTIMATORS, max_depth=RF_MAX_DEPTH)
    # dummy data for initial fit
    X_dummy = np.random.randn(100, 6)
    y_dummy = np.random.randint(0, 2, size=100)
    model.fit(X_dummy, y_dummy)

    # persist for next time
    joblib.dump(model, path)
    logger.info(f"Fallback model trained and saved to {path}")
    return model

# ─── META-LEARNING WEIGHT UPDATER ────────────────────────────────────────────
def update_signal_weights():
    """
    Recompute signal weights via exponential smoothing of recent PnL performance
    and persist to SIGNAL_WEIGHTS_FILE.
    """
    if not os.path.exists(TRADE_LOG_FILE):
        logger.warning("No trades log found; skipping weight update.")
        return

    # load executed trades
    df = pd.read_csv(TRADE_LOG_FILE).dropna(subset=["entry_price", "exit_price", "signal_tags"])
    # compute PnL per tag
    df["pnl"] = (df["exit_price"] - df["entry_price"]) * df["side"].apply(lambda s: 1 if s == "buy" else -1)

    # aggregate by tag
    stats = {}
    for _, row in df.iterrows():
        for tag in row["signal_tags"].split("+"):
            stats.setdefault(tag, []).append(row["pnl"])

    # calculate new raw weights (win rates)
    new_weights = {tag: round(np.mean([1 if p > 0 else 0 for p in pnls]), 3)
                   for tag, pnls in stats.items()}

    # load existing weights (if any) and EMA-smooth
    ALPHA = 0.2
    if os.path.exists(SIGNAL_WEIGHTS_FILE):
        old = pd.read_csv(SIGNAL_WEIGHTS_FILE).set_index("signal")["weight"].to_dict()
    else:
        old = {}

    merged = {}
    for tag, w in new_weights.items():
        merged[tag] = round(ALPHA * w + (1 - ALPHA) * old.get(tag, w), 3)

    # persist updated weights
    out_df = pd.DataFrame.from_dict(merged, orient="index", columns=["weight"]).reset_index()
    out_df.columns = ["signal", "weight"]
    out_df.to_csv(SIGNAL_WEIGHTS_FILE, index=False)
    logger.info(f"[MetaLearn] Updated weights for {len(merged)} signals.")

# ─── MAIN LOOP ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    # — Prometheus & Healthcheck —
    start_http_server(8000)
    threading.Thread(target=start_healthcheck, daemon=True).start()
    logger.info("Healthcheck endpoint running on port 8080")

    # — Daily summary job —
    def daily_summary():
        # TODO: compute & notify
        pass
    schedule.every().day.at("17:30").do(daily_summary)

    # — Model & scheduler —
    model = load_model()
    logger.info("🚀 AI Trading Bot is live!")
    schedule.every(1).minutes.do(lambda: run_all_trades(model))
    schedule.every(6).hours.do(update_signal_weights)

    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            logger.exception(f"Scheduler loop error: {e}")
        time.sleep(30)
