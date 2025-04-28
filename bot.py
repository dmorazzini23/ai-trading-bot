# ─── STANDARD LIBRARIES ─────────────────────────────────────────────────────
import os
import sys
import time
import csv
import re
import threading
import multiprocessing
from datetime import datetime, date, time as dt_time, timedelta
from zoneinfo import ZoneInfo
from functools import lru_cache
from typing import Optional, Tuple
from dataclasses import dataclass, field
from threading import Semaphore

import logging
logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
)

# ─── STRUCTURED LOGGING, RETRIES & RATE LIMITING ────────────────────────────
import structlog
from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential, retry_if_exception_type
from ratelimit import limits, sleep_and_retry

# ─── THIRD-PARTY LIBRARIES ────────────────────────────────────────────────────
import numpy as np
np.NaN = np.nan
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

# for check_daily_loss()
day_start_equity: Optional[Tuple[date, float]] = None
daily_loss: float = 0.0

# ─── A. ENVIRONMENT, SENTRY & LOGGING ────────────────────────────────────────
load_dotenv()
RUN_HEALTH = os.getenv("RUN_HEALTHCHECK", "1") == "1"
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    traces_sample_rate=0.1,
    environment=os.getenv("BOT_MODE", "live"),
)
structlog.configure(logger_factory=structlog.stdlib.LoggerFactory())
logger = structlog.get_logger()

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
    def __init__(self, mode="balanced"):
        self.mode = mode.lower()
        self.params = self.set_parameters()

    def set_parameters(self):
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

    def get_config(self):
        return self.params

BOT_MODE = os.getenv("BOT_MODE", "balanced")
mode     = BotMode(BOT_MODE)
params   = mode.get_config()

# ─── CONFIGURATION CONSTANTS ─────────────────────────────────────────────────
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
RF_ESTIMATORS            = 225
RF_MAX_DEPTH             = 5
ATR_LENGTH               = 12
CONF_THRESHOLD           = params["CONF_THRESHOLD"]
CONFIRMATION_COUNT       = params["CONFIRMATION_COUNT"]
KELLY_FRACTION           = params["KELLY_FRACTION"]
CAPITAL_CAP              = params["CAPITAL_CAP"]
PACIFIC                  = ZoneInfo("America/Los_Angeles")

# ─── ASSERT ALPACA KEYS ───────────────────────────────────────────────────────
ALPACA_API_KEY    = os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
ALPACA_BASE_URL   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    logger.error("❌ Missing Alpaca API credentials; please check .env")
    sys.exit(1)

# ─── TYPED EXCEPTION ──────────────────────────────────────────────────────────
class DataFetchError(Exception):
    pass

# ─── BOT CONTEXT ─────────────────────────────────────────────────────────────
@dataclass
class BotContext:
    api: REST
    data_fetcher: "DataFetcher"
    signal_manager: "SignalManager"
    trade_logger: "TradeLogger"
    sem: Semaphore
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

# ─── CORE CLASSES ─────────────────────────────────────────────────────────────
class DataFetcher:
    @lru_cache(maxsize=None)
    def get_daily_df(self, ctx: BotContext, symbol: str) -> Optional[pd.DataFrame]:
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

class TradeLogger:
    def __init__(self, path: str = TRADE_LOG_FILE):
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
            rows = list(csv.reader(f))
            header, data = rows[0], rows[1:]
            for row in data:
                if row[0] == symbol and row[3] == "":
                    entry_t = datetime.fromisoformat(row[1])
                    days = (datetime.utcnow() - entry_t).days
                    cls = ("day_trade" if days == 0
                           else "swing_trade" if days < 5
                           else "long_trade")
                    row[3], row[4], row[8] = datetime.utcnow().isoformat(), exit_price, cls
                    break
            f.seek(0)
            f.truncate()
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(data)

class SignalManager:
    def __init__(self):
        self.momentum_lookback = 5
        self.mean_rev_lookback = 20
        self.mean_rev_zscore_threshold = 1.5
        self.regime_volatility_threshold = REGIME_ATR_THRESHOLD

    def signal_momentum(self, df):
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
        try:
            ma = df['Close'].rolling(self.mean_rev_lookback).mean()
            sd = df['Close'].rolling(self.mean_rev_lookback).std()
            df['zscore'] = (df['Close'] - ma) / sd
            val = df['zscore'].iloc[-1]
            signal = (0 if val > self.mean_rev_zscore_threshold
                      else 1 if val < -self.mean_rev_zscore_threshold
                      else -1)
            weight = min(abs(val) / 3, 1.0)
            return signal, weight, 'mean_reversion'
        except Exception:
            logger.exception("Error in signal_mean_reversion")
            return -1, 0.0, 'mean_reversion'

    def signal_ml(self, df, model):
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
        try:
            val = fetch_sentiment(ticker)
            signal = 1 if val > 0 else 0 if val < 0 else -1
            weight = min(abs(val), 0.2)
            return signal, weight, 'sentiment'
        except Exception:
            logger.exception("Error in signal_sentiment")
            return -1, 0.0, 'sentiment'

    def signal_regime(self):
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
        try:
            val = df['stochrsi'].iloc[-1]
            signal = 1 if val < 0.2 else 0 if val > 0.8 else -1
            return signal, 0.3, 'stochrsi'
        except Exception:
            logger.exception("Error in signal_stochrsi")
            return -1, 0.0, 'stochrsi'

    def signal_obv(self, df):
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
        try:
            body = abs(df['Close'] - df['Open'])
            vsa = df['Volume'] * body
            score = vsa.iloc[-1]
            avg = vsa.rolling(20).mean().iloc[-1]
            signal = (1 if df['Close'].iloc[-1] > df['Open'].iloc[-1]
                      else 0 if df['Close'].iloc[-1] < df['Open'].iloc[-1]
                      else -1)
            weight = min(score / avg, 1.0)
            return signal, weight, 'vsa'
        except Exception:
            logger.exception("Error in signal_vsa")
            return -1, 0.0, 'vsa'

    def load_signal_weights(self):
        if not os.path.exists(SIGNAL_WEIGHTS_FILE):
            return {}
        df = pd.read_csv(SIGNAL_WEIGHTS_FILE)
        return {row['signal']: row['weight'] for _, row in df.iterrows()}

    def evaluate(self, df, ticker, model):
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

# ─── GLOBAL STATE ─────────────────────────────────────────────────────────────
data_fetcher   = DataFetcher()
signal_manager = SignalManager()
trade_logger   = TradeLogger()
semaphore      = threading.Semaphore(4)
api            = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)

ctx = BotContext(
    api=api,
    data_fetcher=data_fetcher,
    signal_manager=signal_manager,
    trade_logger=trade_logger,
    sem=semaphore,
    volume_threshold=VOLUME_THRESHOLD,
    entry_start_offset=ENTRY_START_OFFSET,
    entry_end_offset=ENTRY_END_OFFSET,
    market_open=MARKET_OPEN,
    market_close=MARKET_CLOSE,
    regime_lookback=REGIME_LOOKBACK,
    regime_atr_threshold=REGIME_ATR_THRESHOLD,
    daily_loss_limit=DAILY_LOSS_LIMIT,
)

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
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise DataFetchError(f"No data for {symbol}")
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
    text = " ".join(a["title"] + " " + a.get("description", "") for a in articles).lower()
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

# ─── CHECKS & GUARDS ─────────────────────────────────────────────────────────
@sleep_and_retry
@limits(calls=200, period=60)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type(APIError)
)
def check_daily_loss() -> bool:
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
        daily_drawdown.set(0.0)
        return False
    loss = (day_start_equity[1] - cash) / day_start_equity[1]
    daily_drawdown.set(loss)
    return loss >= DAILY_LOSS_LIMIT

def check_halt_flag() -> bool:
    if not os.path.exists(HALT_FLAG_PATH):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(HALT_FLAG_PATH))
    return age < timedelta(hours=1)

def within_market_hours() -> bool:
    now = now_pacific()
    start = datetime.combine(now.date(), MARKET_OPEN, PACIFIC) + ENTRY_START_OFFSET
    end   = datetime.combine(now.date(), MARKET_CLOSE, PACIFIC) - ENTRY_END_OFFSET
    return start <= now <= end

@retry(stop=stop_after_attempt(3), wait=wait_fixed(0.5))
def check_market_regime() -> bool:
    df = fetch_data(ctx, "SPY", period="1mo", interval="1d")
    if df is None or df.empty:
        logger.warning("[check_market_regime] No SPY data – failing regime check")
        return False
    for col in ["High", "Low", "Close"]:
        if col not in df.columns:
            logger.warning(f"[check_market_regime] Missing column '{col}' in SPY data")
            return False
    df.dropna(subset=["High", "Low", "Close"], inplace=True)
    if len(df) < REGIME_LOOKBACK:
        logger.warning(f"[check_market_regime] Not enough SPY rows after cleaning: {len(df)} – need {REGIME_LOOKBACK}")
        return False
    try:
        atr_series = ta.atr(df["High"], df["Low"], df["Close"], length=REGIME_LOOKBACK)
        atr_val = atr_series.iloc[-1]
        if pd.isna(atr_val):
            logger.warning("[check_market_regime] ATR value is NaN – failing regime check")
            return False
    except Exception as e:
        logger.warning(f"[check_market_regime] ATR calc failed: {e}")
        return False
    vol = df["Close"].pct_change().std()
    return (atr_val <= REGIME_ATR_THRESHOLD) or (vol <= 0.015)

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
        d = fetch_data(ctx, s, period="3mo", interval="1d")
        rets[s] = d["Close"].pct_change().dropna()
    min_len = min(len(r) for r in rets.values())
    mat = pd.DataFrame({s: rets[s].tail(min_len) for s in open_syms})
    corr_matrix = mat.corr().abs()
    avg_corr = corr_matrix.where(~np.eye(len(open_syms), dtype=bool)).stack().mean()
    return avg_corr > CORRELATION_THRESHOLD

def now_pacific() -> datetime:
    return datetime.now(PACIFIC)

def is_within_entry_window(ctx: BotContext) -> bool:
    now = now_pacific()
    start = (datetime.combine(now.date(), ctx.market_open) + ctx.entry_start_offset).time()
    end   = (datetime.combine(now.date(), ctx.market_close) - ctx.entry_end_offset).time()
    if not (start <= now.time() <= end):
        logger.info(f"[SKIP] entry window ({start}–{end}), now={now.time()}")
        return False
    return True

# ─── SIZING & EXECUTION ───────────────────────────────────────────────────────
def fractional_kelly_size(
    balance: float,
    price: float,
    atr: float,
    win_prob: float,
    payoff_ratio: float = 1.5
) -> int:
    # ensure peak equity file exists & update peak if new high
    if not os.path.exists(PEAK_EQUITY_FILE):
        with portalocker.Lock(PEAK_EQUITY_FILE, 'w', timeout=1) as f:
            f.write(str(balance))
        peak_equity = balance
    else:
        with portalocker.Lock(PEAK_EQUITY_FILE, 'r+', timeout=1) as f:
            content = f.read().strip()
            peak_equity = float(content) if content else balance
            if balance > peak_equity:
                f.seek(0)
                f.truncate()
                f.write(str(balance))
                peak_equity = balance

    # compute drawdown and choose fraction dynamically
    drawdown = (peak_equity - balance) / peak_equity
    if drawdown > 0.10:
        frac = 0.3
    elif drawdown > 0.05:
        frac = 0.45
    else:
        frac = KELLY_FRACTION  # read the up-to-date global

    # Kelly calculation
    edge = win_prob - (1 - win_prob) / payoff_ratio
    kelly = max(edge / payoff_ratio, 0) * frac

    # dollars to risk
    dollars_to_risk = kelly * balance
    if atr <= 0:
        return 1

    # position sizing capped by ATR, capital cap, and max size
    raw_pos = dollars_to_risk / atr
    cap_pos = (balance * CAPITAL_CAP) / price
    size = int(min(raw_pos, cap_pos, MAX_POSITION_SIZE))

    return max(size, 1)

def submit_order(
    ctx: BotContext,
    ticker: str,
    qty: int,
    side: str,
    order_type_override: str = None
) -> bool:
    orders_total.inc()
    order_type = order_type_override or ORDER_TYPE

    try:
        with ctx.sem:
            # fetch the latest quote (handle both v1 & v2 SDKs)
            try:
                quote = ctx.api.get_last_quote(ticker)
            except AttributeError:
                quote = ctx.api.get_latest_quote(ticker)

            if order_type == "limit":
                bid, ask = quote.bidprice, quote.askprice
                spread = (ask - bid) if (ask and bid) else 0
                limit_price = (bid + 0.25 * spread) if side == "buy" else (ask - 0.25 * spread)
                ctx.api.submit_order(
                    symbol=ticker,
                    qty=qty,
                    side=side,
                    type="limit",
                    limit_price=round(limit_price, 2),
                    time_in_force="gtc"
                )
                expected = round(limit_price, 2)
            else:
                # for market orders, use the quote to log expected fill
                expected = quote.askprice if side == "buy" else quote.bidprice
                ctx.api.submit_order(
                    symbol=ticker,
                    qty=qty,
                    side=side,
                    type="market",
                    time_in_force="gtc"
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
                time_in_force="gtc"
            )
            return True

        logger.warning(f"[submit_order] APIError: {e}")
        raise

    except Exception:
        order_failures.inc()
        logger.exception(f"[submit_order] unexpected error for {ticker}")
        raise

    except APIError as e:
        # if requested size exceeds available, submit what we can
        m = re.search(r"requested: (\d+), available: (\d+)", str(e))
        if m and int(m.group(2)) > 0:
            available = int(m.group(2))
            ctx.api.submit_order(
                symbol=ticker,
                qty=available,
                side=side,
                type=order_type,
                time_in_force="gtc"
            )
            return True

        logger.warning(f"[submit_order] APIError: {e}")
        raise

    except Exception:
        order_failures.inc()
        logger.exception(f"[submit_order] unexpected error for {ticker}")
        raise

def update_trailing_stop(
    ctx: BotContext,
    ticker: str,
    price: float,
    qty: int,
    atr: float,
    factor: float = TRAILING_FACTOR
) -> str:
    te = ctx.trailing_extremes

    if qty > 0:
        # record new high for a long position
        te[ticker] = max(te.get(ticker, price), price)
        # exit if price has fallen below the trailing stop
        if price < te[ticker] - factor * atr:
            return "exit_long"

    elif qty < 0:
        # record new low for a short position
        te[ticker] = min(te.get(ticker, price), price)
        # exit if price has risen above the trailing stop
        if price > te[ticker] + factor * atr:
            return "exit_short"

    return "hold"

def calculate_entry_size(
    ctx: BotContext, symbol: str, price: float, atr: float, win_prob: float
) -> int:
    return fractional_kelly_size(
        balance=float(ctx.api.get_account().cash),
        price=price, atr=atr, win_prob=win_prob
    )

def execute_entry(ctx: BotContext, symbol: str, qty: int, side: str) -> None:
    # submit the order
    submit_order(ctx, symbol, qty, side)

    # get the latest price from minute data
    df_min = ctx.data_fetcher.get_minute_df(ctx, symbol)
    price = df_min["Close"].iloc[-1]

    # log the trade entry
    ctx.trade_logger.log_entry(
        symbol=symbol,
        price=price,
        qty=qty,
        side=side,
        strategy="",
        signal_tags=""
    )

    # set take-profit target: X% above (or below) entry price
    if side == "buy":
        tp = price * (1 + TAKE_PROFIT_FACTOR)
    else:
        tp = price * (1 - TAKE_PROFIT_FACTOR)

    ctx.take_profit_targets[symbol] = tp

def execute_exit(ctx: BotContext, symbol: str, qty: int) -> None:
    submit_order(ctx, symbol, qty, "sell" if qty>0 else "buy")
    ctx.trade_logger.log_exit(
        symbol,
        ctx.data_fetcher.get_minute_df(ctx, symbol)["Close"].iloc[-1]
    )
    ctx.take_profit_targets.pop(symbol, None)

# ─── SIGNAL & TRADE LOGIC ────────────────────────────────────────────────────
def signal_and_confirm(ctx: BotContext, symbol: str, df: pd.DataFrame, model) -> Tuple[int,float,str]:
    sig, conf, strat = ctx.signal_manager.evaluate(df, symbol, model)
    if sig == -1 or conf < CONF_THRESHOLD:
        logger.info(f"[SKIP] {symbol} no/low signal (sig={sig},conf={conf:.2f})")
        return -1, 0.0, ""
    return sig, conf, strat

def pre_trade_checks(ctx: BotContext, symbol: str, balance: float) -> bool:
    if check_halt_flag():
        logger.info(f"[SKIP] HALT_FLAG – {symbol}")
        return False
    if not within_market_hours():
        logger.info(f"[SKIP] Market closed – {symbol}")
        return False
    if check_daily_loss():
        logger.info(f"[SKIP] Daily-loss limit – {symbol}")
        return False
    if not check_market_regime():
        logger.info(f"[SKIP] Market regime – {symbol}")
        return False
    if too_many_positions():
        logger.info(f"[SKIP] Max positions – {symbol}")
        return False
    if too_correlated(symbol):
        logger.info(f"[SKIP] Correlation – {symbol}")
        return False
    return ctx.data_fetcher.get_daily_df(ctx, symbol) is not None

def should_enter(ctx: BotContext, symbol: str, balance: float) -> bool:
    return pre_trade_checks(ctx, symbol, balance) and is_within_entry_window(ctx)

def should_exit(ctx: BotContext, symbol: str, price: float, atr: float) -> Tuple[bool,int,str]:
    try:
        pos = int(ctx.api.get_position(symbol).qty)
    except Exception:
        pos = 0
    tp = ctx.take_profit_targets.get(symbol)
    if pos and tp and ((pos>0 and price>=tp) or (pos<0 and price<=tp)):
        return True, int(abs(pos)*SCALING_FACTOR), "take_profit"
    action = update_trailing_stop(ctx, symbol, price, pos, atr)
    if action=="exit_long" and pos>0:
        return True, pos, "trailing_stop"
    if action=="exit_short" and pos<0:
        return True, abs(pos), "trailing_stop"
    return False, 0, ""

def trade_logic(ctx: BotContext, symbol: str, balance: float, model) -> None:
    # only proceed if all pre‐trade checks pass
    if not should_enter(ctx, symbol, balance):
        return

    # fetch minute data and bail out if missing
    raw = ctx.data_fetcher.get_minute_df(ctx, symbol)
    if raw is None or raw.empty:
        logger.info(f"[SKIP] no minute data for {symbol}")
        return

    # prepare indicators on valid data
    df = prepare_indicators(raw)

    # get and confirm signal
    sig, conf, _ = signal_and_confirm(ctx, symbol, df, model)
    if sig == -1:
        return

    price, atr = df["Close"].iloc[-1], df["atr"].iloc[-1]

    # check for exit conditions first
    do_exit, qty, _ = should_exit(ctx, symbol, price, atr)
    if do_exit:
        execute_exit(ctx, symbol, qty)
        return

    # size the new entry
    size = calculate_entry_size(ctx, symbol, price, atr, conf)
    if size > 0:
        try:
            pos = int(ctx.api.get_position(symbol).qty)
        except Exception:
            pos = 0
        # submit entry order
        execute_entry(
            ctx,
            symbol,
            size + abs(pos),
            "buy" if sig == 1 else "sell"
        )

def run_all_trades(model):
    logger.info(f"🔄 run_all_trades fired at {datetime.utcnow().isoformat()}")
    if check_halt_flag():
        logger.info("Trading halted via HALT_FLAG_FILE.")
        return
    try:
        acct = api.get_account()
        current_cash = float(acct.cash)
    except Exception:
        logger.error("Failed to retrieve account balance.")
        return
    if os.path.exists(EQUITY_FILE):
        with open(EQUITY_FILE, "r") as f:
            last_cash = float(f.read().strip())
    else:
        last_cash = current_cash
    global KELLY_FRACTION
    KELLY_FRACTION = 0.7 if current_cash > last_cash * 1.03 else params["KELLY_FRACTION"]
    with open(EQUITY_FILE, "w") as f:
        f.write(str(current_cash))
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

# ─── UTILITIES ────────────────────────────────────────────────────────────────
def load_model(path: str = MODEL_PATH):
    if os.path.exists(path):
        logger.info(f"Loading trained model from {path}")
        return joblib.load(path)
    logger.info("No model found; training fallback RandomForestClassifier")
    model = RandomForestClassifier(n_estimators=RF_ESTIMATORS, max_depth=RF_MAX_DEPTH)
    X_dummy = np.random.randn(100, 6)
    y_dummy = np.random.randint(0, 2, size=100)
    model.fit(X_dummy, y_dummy)
    joblib.dump(model, path)
    logger.info(f"Fallback model trained and saved to {path}")
    return model

def update_signal_weights():
    if not os.path.exists(TRADE_LOG_FILE):
        logger.warning("No trades log found; skipping weight update.")
        return
    df = pd.read_csv(TRADE_LOG_FILE).dropna(subset=["entry_price","exit_price","signal_tags"])
    df["pnl"] = (df["exit_price"] - df["entry_price"]) * df["side"].apply(lambda s: 1 if s=="buy" else -1)
    stats = {}
    for _, row in df.iterrows():
        for tag in row["signal_tags"].split("+"):
            stats.setdefault(tag, []).append(row["pnl"])
    new_weights = {tag: round(np.mean([1 if p>0 else 0 for p in pnls]),3) for tag,pnls in stats.items()}
    ALPHA = 0.2
    old = pd.read_csv(SIGNAL_WEIGHTS_FILE).set_index("signal")["weight"].to_dict() if os.path.exists(SIGNAL_WEIGHTS_FILE) else {}
    merged = {tag: round(ALPHA*w + (1-ALPHA)*old.get(tag,w),3) for tag, w in new_weights.items()}
    out_df = pd.DataFrame.from_dict(merged, orient="index", columns=["weight"]).reset_index()
    out_df.columns = ["signal","weight"]
    out_df.to_csv(SIGNAL_WEIGHTS_FILE, index=False)
    logger.info(f"[MetaLearn] Updated weights for {len(merged)} signals.")

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
    win_rates = {tag: round(np.mean([1 if p>0 else 0 for p in pnls]),3)
                 for tag,pnls in results.items() if len(pnls)>=min_trades}
    filtered = {tag:wr for tag,wr in win_rates.items() if wr>=threshold}
    logger.info(f"[MetaLearn] Keeping signals: {list(filtered.keys()) or 'none'}")
    return filtered

def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ensure our timestamp column is named "Date"
    if df.index.name:
        df = df.reset_index().rename(columns={df.index.name: "Date"})
    else:
        df = df.reset_index().rename(columns={"index": "Date"})
    df.sort_values("Date", inplace=True)

    df.reset_index(drop=True, inplace=True)
    df["vwap"]    = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"])
    df["sma_50"]  = ta.sma(df["Close"], length=50)
    df["sma_200"] = ta.sma(df["Close"], length=200)
    df["rsi"]     = ta.rsi(df["Close"], length=14)

    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    df["macd"], df["macds"] = macd["MACD_12_26_9"], macd["MACDs_12_26_9"]

    df["atr"]     = ta.atr(df["High"], df["Low"], df["Close"], length=ATR_LENGTH)

    ich = ta.ichimoku(df["High"], df["Low"], df["Close"])
    df["ichimoku_base"], df["ichimoku_conv"] = ich["ISA_9"], ich["ISB_26"]

    df["stochrsi"] = ta.stochrsi(df["Close"])["STOCHRSIk_14_14_3_3"]

    df.ffill().bfill(inplace=True)
    df.dropna(subset=[
        "vwap", "sma_50", "sma_200", "rsi",
        "macd", "macds", "atr",
        "ichimoku_base", "ichimoku_conv", "stochrsi"
    ], inplace=True)

    return df

def load_tickers(path: str = TICKERS_FILE) -> list[str]:
    tickers = []
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

# ─── MAIN LOOP ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    # Prometheus metrics
    start_http_server(8000)
    logger.info("Prometheus metrics server started on port 8000")
    
    # Healthcheck endpoint (only if RUN_HEALTH is true)
    if RUN_HEALTH:
        threading.Thread(target=start_healthcheck, daemon=True).start()
        logger.info("Healthcheck endpoint running on port 8080")

    # End-of-day summary
    def daily_summary():
        # TODO: load trades.csv → compute PnL, drawdown, win rate → post to Slack/email
        pass
    schedule.every().day.at("17:30").do(daily_summary)
    logger.info("Scheduled daily summary at 17:30")

    # Load model & schedule jobs
    model = load_model()
    logger.info("🚀 AI Trading Bot is live!")
    schedule.every(1).minutes.do(lambda: run_all_trades(model))
    schedule.every(6).hours.do(update_signal_weights)
    logger.info("Scheduled run_all_trades every 1 minute and update_signal_weights every 6 hours")

    # Scheduler loop
    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            logger.exception(f"Scheduler loop error: {e}")
        time.sleep(30)
