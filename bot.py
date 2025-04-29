# ─── STANDARD LIBRARIES ─────────────────────────────────────────────────────
import os
import csv
import re
import time
from datetime import datetime, date, time as dt_time, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
from threading import Semaphore, Thread
from concurrent.futures import ThreadPoolExecutor, as_completed

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
import pandas as pd
from numpy import nan as npNaN
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
log = logging.getLogger(__name__)

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

    def set_parameters(self) -> Dict[str, float]:
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

    def get_config(self) -> Dict[str, float]:
        return self.params

BOT_MODE = os.getenv("BOT_MODE", "balanced")
mode     = BotMode(BOT_MODE)
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
    confirmation_count: Dict[str, int] = field(default_factory=dict)
    trailing_extremes: Dict[str, float] = field(default_factory=dict)
    take_profit_targets: Dict[str, float] = field(default_factory=dict)

# ─── CORE CLASSES ─────────────────────────────────────────────────────────────
class DataFetcher:
    def __init__(self) -> None:
        self._daily_cache  : Dict[str, Optional[pd.DataFrame]] = {}
        self._minute_cache : Dict[str, Optional[pd.DataFrame]] = {}

    def get_daily_df(self, ctx: BotContext, symbol: str) -> Optional[pd.DataFrame]:
        if symbol not in self._daily_cache:
            try:
                df = fetch_data(ctx, symbol, period="5d", interval="1d")
            except DataFetchError:
                logger.info(f"[SKIP] No daily data for {symbol}")
                df = None
            self._daily_cache[symbol] = df
        return self._daily_cache[symbol]

    def get_minute_df(self, ctx: BotContext, symbol: str) -> Optional[pd.DataFrame]:
        if symbol not in self._minute_cache:
            try:
                df = fetch_data(ctx, symbol, period="1d", interval="1m")
            except DataFetchError:
                logger.info(f"[SKIP] No minute data for {symbol}")
                df = None
            self._minute_cache[symbol] = df
        return self._minute_cache[symbol]

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
        now = datetime.utcnow().isoformat()
        with portalocker.Lock(self.path, 'a', timeout=1) as f:
            csv.writer(f).writerow([symbol, now, price, "","", qty, side, strategy, "", signal_tags])

    def log_exit(self, symbol: str, exit_price: float) -> None:
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
        try:
            df['momentum'] = df['Close'].pct_change(self.momentum_lookback)
            if 'momentum' not in df or df['momentum'].empty:
                return -1, 0.0, 'momentum'
            val = df['momentum'].iloc[-1]
            signal = 1 if val > 0 else 0 if val < 0 else -1
            weight = min(abs(val) * 10, 1.0)
            return signal, weight, 'momentum'
        except Exception:
            logger.exception("Error in signal_momentum")
            return -1, 0.0, 'momentum'

    def signal_mean_reversion(self, df: pd.DataFrame) -> Tuple[int, float, str]:
        try:
            ma = df['Close'].rolling(self.mean_rev_lookback).mean()
            sd = df['Close'].rolling(self.mean_rev_lookback).std()
            df['zscore'] = (df['Close'] - ma) / sd
            if 'zscore' not in df or df['zscore'].empty:
                return -1, 0.0, 'mean_reversion'
            val = df['zscore'].iloc[-1]
            signal = (0 if val > self.mean_rev_zscore_threshold
                      else 1 if val < -self.mean_rev_zscore_threshold
                      else -1)
            weight = min(abs(val)/3, 1.0)
            return signal, weight, 'mean_reversion'
        except Exception:
            logger.exception("Error in signal_mean_reversion")
            return -1, 0.0, 'mean_reversion'

    def signal_ml(self, df: pd.DataFrame, model) -> Tuple[int, float, str]:
        try:
            feats = ['rsi','sma_50','sma_200','macd','macds','atr']
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

    def signal_sentiment(self, ctx: BotContext, ticker: str) -> Tuple[int, float, str]:
        try:
            val = fetch_sentiment(ctx, ticker)
            sig = 1 if val > 0 else 0 if val < 0 else -1
            weight = min(abs(val), 0.2)
            return sig, weight, 'sentiment'
        except Exception:
            logger.exception("Error in signal_sentiment")
            return -1, 0.0, 'sentiment'

    def signal_regime(self, ctx: BotContext) -> Tuple[int, float, str]:
        try:
            df = fetch_data(ctx, "SPY", period=f"{ctx.regime_lookback}d", interval="1d")
            if df is None or df.empty:
                return -1, 0.0, 'regime'
            atr_series = ta.atr(df["High"], df["Low"], df["Close"], length=ctx.regime_lookback)
            atr_val = atr_series.iloc[-1]
            if pd.isna(atr_val):
                return -1, 0.0, 'regime'
            sig = 1 if atr_val < ctx.regime_atr_threshold else 0
            return sig, 0.6, 'regime'
        except Exception:
            logger.exception("Error in signal_regime")
            return -1, 0.0, 'regime'

    def signal_stochrsi(self, df: pd.DataFrame) -> Tuple[int, float, str]:
        try:
            if 'stochrsi' not in df or df['stochrsi'].empty:
                return -1, 0.0, 'stochrsi'
            val = df['stochrsi'].iloc[-1]
            signal = 1 if val < 0.2 else 0 if val > 0.8 else -1
            return signal, 0.3, 'stochrsi'
        except Exception:
            logger.exception("Error in signal_stochrsi")
            return -1, 0.0, 'stochrsi'

    def signal_obv(self, df: pd.DataFrame) -> Tuple[int, float, str]:
        try:
            obv = ta.obv(df['Close'], df['Volume'])
            last5 = obv.tail(5)
            if len(last5) < 2:
                return -1, 0.0, 'obv'
            slope = np.polyfit(range(len(last5)), last5, 1)[0]
            signal = 1 if slope > 0 else 0 if slope < 0 else -1
            weight = min(abs(slope)/1e6, 1.0)
            return signal, weight, 'obv'
        except Exception:
            logger.exception("Error in signal_obv")
            return -1, 0.0, 'obv'

    def signal_vsa(self, df: pd.DataFrame) -> Tuple[int, float, str]:
        try:
            body = abs(df['Close'] - df['Open'])
            vsa = df['Volume'] * body
            if vsa.empty:
                return -1, 0.0, 'vsa'
            score = vsa.iloc[-1]
            avg_series = vsa.rolling(20).mean()
            avg = avg_series.iloc[-1] if not avg_series.empty else np.nan
            if df['Close'].empty or df['Open'].empty:
                return -1, 0.0, 'vsa'
            signal = 1 if df['Close'].iloc[-1] > df['Open'].iloc[-1] \
                     else 0 if df['Close'].iloc[-1] < df['Open'].iloc[-1] \
                     else -1
            weight = min(score/avg, 1.0) if avg and not np.isnan(avg) else 0.0
            return signal, weight, 'vsa'
        except Exception:
            logger.exception("Error in signal_vsa")
            return -1, 0.0, 'vsa'

    def load_signal_weights(self) -> Dict[str, float]:
        if not os.path.exists(SIGNAL_WEIGHTS_FILE):
            return {}
        df = pd.read_csv(SIGNAL_WEIGHTS_FILE)
        return {row['signal']: row['weight'] for _, row in df.iterrows()}

    def evaluate(self, ctx: BotContext, df: pd.DataFrame, ticker: str, model) -> Tuple[int, float, str]:
        signals: List[Tuple[int, float, str]] = []
        allowed_tags = set(load_global_signal_performance() or [])
        weights = self.load_signal_weights()

        fns = [
            self.signal_momentum,
            self.signal_mean_reversion,
            lambda d: self.signal_ml(d, model),
            lambda d: self.signal_sentiment(ctx, ticker),
            lambda d: self.signal_regime(ctx),
            self.signal_stochrsi,
            self.signal_obv,
            self.signal_vsa,
        ]

        for fn in fns:
            try:
                s, w, lab = fn(df)
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

# ─── WRAPPED I/O CALLS ───────────────────────────────────────────────────────
@sleep_and_retry
@limits(calls=60, period=60)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type((requests.RequestException, DataFetchError))
)
def fetch_data(ctx, symbol, period="1mo", interval="1d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, progress=False)

    # reset whatever index → “Date”
    df = df.copy()
    if df.index.name:
        df = df.reset_index().rename(columns={df.index.name:"Date"})
    else:
        df = df.reset_index().rename(columns={"index":"Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    log.info(f"fetch_data raw columns: {df.columns.tolist()}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        log.info(f"after collapsing MultiIndex, columns: {df.columns.tolist()}")

    # drop any duplicate column names (e.g. double "Close")
    df = df.loc[:, ~df.columns.duplicated()]

    missing = [c for c in ("High","Low","Close") if c not in df.columns]
    if missing:
        log.error(f"fetch_data: missing required columns {missing}, got {df.columns.tolist()}")
        raise KeyError(f"Missing data columns: {missing}")

    df.dropna(subset=["High","Low","Close"], inplace=True)
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
            headers={"User-Agent":"AI Trading Bot"}
        )
        r.raise_for_status()

    soup = BeautifulSoup(r.content, "lxml")

    # 1) debug: how many <tr> rows did we find?
    rows = soup.find_all("tr")
    log.info(f"get_sec_headlines({ticker}) ── total <tr> rows: {len(rows)}")

    texts = []
    for a in soup.find_all("a"):
        if "8-K" in a.text:
            tr = a.find_parent("tr")
            if not tr:
                log.warning("found an <a> with '8-K' but no parent <tr>")
                continue

            tds = tr.find_all("td")
            # 2) guard: make sure the cell we want exists
            if len(tds) < 4:
                log.warning(f"unexpected <td> count {len(tds)} in row, skipping")
                continue

            texts.append(tds[-1].get_text(strip=True))

    if not texts:
        log.warning(f"get_sec_headlines({ticker}) found no 8-K entries")
    return " ".join(texts)

@limits(calls=60, period=60)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type((requests.RequestException, DataFetchError))
)
def fetch_sentiment(ctx, ticker) -> float:
    """
    Fetches news sentiment for `ticker` via NewsAPI.
    Returns a float in [-1.0, +1.0], or 0.0 on rate-limit (429).
    """
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
            log.warning(f"fetch_sentiment({ticker}) rate-limited → returning neutral 0.0")
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
        acct = api.get_account()
        cash = float(acct.cash)
        log.debug(f"account cash: {cash}")
    except Exception as e:
        log.warning(f"[check_daily_loss] could not fetch account cash: {e!r}")
        return False

    today = date.today()
    if day_start_equity is None or day_start_equity[0] != today:
        day_start_equity = (today, cash)
        daily_drawdown.set(0.0)
        log.info(f"reset day_start_equity to {cash} on {today}")
        return False

    loss = (day_start_equity[1] - cash) / day_start_equity[1]
    daily_drawdown.set(loss)
    log.info(f"daily drawdown is {loss:.2%}")
    return loss >= DAILY_LOSS_LIMIT


def check_halt_flag() -> bool:
    if not os.path.exists(HALT_FLAG_PATH):
        return False
    age = datetime.now()-datetime.fromtimestamp(os.path.getmtime(HALT_FLAG_PATH))
    return age<timedelta(hours=1)


def within_market_hours() -> bool:
    now = now_pacific()
    start = datetime.combine(now.date(),MARKET_OPEN,PACIFIC)+ENTRY_START_OFFSET
    end   = datetime.combine(now.date(),MARKET_CLOSE,PACIFIC)-ENTRY_END_OFFSET
    return start<=now<=end

@retry(stop=stop_after_attempt(3), wait=wait_fixed(0.5))
def check_market_regime() -> bool:
    df = fetch_data(ctx, "SPY", period="1mo", interval="1d")
    if df is None or df.empty:
        logger.warning("[check_market_regime] No SPY data – failing regime check")
        return False

    # guard columns
    for col in ["High", "Low", "Close"]:
        if col not in df.columns:
            logger.warning(f"[check_market_regime] Missing column '{col}' in SPY data")
            return False

    df.dropna(subset=["High", "Low", "Close"], inplace=True)
    if len(df) < REGIME_LOOKBACK:
        logger.warning(
            f"[check_market_regime] Not enough SPY rows after cleaning: {len(df)} – need {REGIME_LOOKBACK}"
        )
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
        return len(api.list_positions())>=MAX_PORTFOLIO_POSITIONS
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
        if d is None or d.empty:
            # skip symbols without data
            continue
        series = d["Close"].pct_change().dropna()
        if len(series) > 0:
            rets[s] = series

    # need at least two symbols with data to compute correlation
    if len(rets) < 2:
        return False

    # find the minimum length across return series
    min_len = min(len(r) for r in rets.values())
    if min_len < 1:
        return False

    # filter series by min_len and build DataFrame with explicit index
    good_syms = [s for s, r in rets.items() if len(r) >= min_len]
    idx = rets[good_syms[0]].tail(min_len).index
    mat = pd.DataFrame({s: rets[s].tail(min_len).values for s in good_syms}, index=idx)
    corr_matrix = mat.corr().abs()
    # compute average off-diagonal correlation
    avg_corr = corr_matrix.where(~np.eye(len(good_syms), dtype=bool)).stack().mean()
    return avg_corr > CORRELATION_THRESHOLD

def now_pacific() -> datetime:
    return datetime.now(PACIFIC)

def is_within_entry_window(ctx: BotContext) -> bool:
    now = now_pacific()
    start = (datetime.combine(now.date(),ctx.market_open)+ctx.entry_start_offset).time()
    end   = (datetime.combine(now.date(),ctx.market_close)-ctx.entry_end_offset).time()
    if not (start<=now.time()<=end):
        logger.info(f"[SKIP] entry window ({start}–{end}), now={now.time()}")
        return False
    return True

# ─── SIZING & EXECUTION ───────────────────────────────────────────────────────
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
            try:
                quote = ctx.api.get_last_quote(ticker)
            except AttributeError:
                quote = ctx.api.get_latest_quote(ticker)

            if order_type=="limit":
                bid, ask = quote.bidprice, quote.askprice
                spread = (ask-bid) if (ask and bid) else 0
                limit_price = (bid+0.25*spread) if side=="buy" else (ask-0.25*spread)
                ctx.api.submit_order(
                    symbol=ticker, qty=qty, side=side,
                    type="limit", limit_price=round(limit_price,2),
                    time_in_force="gtc"
                )
                expected = round(limit_price,2)
            else:
                expected = quote.askprice if side=="buy" else quote.bidprice
                ctx.api.submit_order(
                    symbol=ticker, qty=qty, side=side,
                    type="market", time_in_force="gtc"
                )

        logger.info(f"[SLIPPAGE] {ticker} expected={expected} side={side} qty={qty}")
        return True

    except APIError as e:
        m = re.search(r"requested: (\d+), available: (\d+)", str(e))
        if m and int(m.group(2))>0:
            available = int(m.group(2))
            ctx.api.submit_order(
                symbol=ticker, qty=available,
                side=side, type=order_type, time_in_force="gtc"
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
    ctx: BotContext, symbol: str, price: float, atr: float, win_prob: float
) -> int:
    return fractional_kelly_size(ctx, float(ctx.api.get_account().cash), price, atr, win_prob)

def execute_entry(ctx: BotContext, symbol: str, qty: int, side: str) -> None:
    submit_order(ctx, symbol, qty, side)
    df_min = ctx.data_fetcher.get_minute_df(ctx, symbol)
    price = df_min["Close"].iloc[-1]
    ctx.trade_logger.log_entry(symbol, price, qty, side, "", "")
    tp = price*(1+TAKE_PROFIT_FACTOR) if side=="buy" else price*(1-TAKE_PROFIT_FACTOR)
    ctx.take_profit_targets[symbol] = tp

def execute_exit(ctx: BotContext, symbol: str, qty: int) -> None:
    submit_order(ctx, symbol, qty, "sell" if qty>0 else "buy")
    ctx.trade_logger.log_exit(symbol, ctx.data_fetcher.get_minute_df(ctx, symbol)["Close"].iloc[-1])
    ctx.take_profit_targets.pop(symbol, None)

# ─── SIGNAL & TRADE LOGIC ────────────────────────────────────────────────────
def signal_and_confirm(ctx: BotContext, symbol: str, df: pd.DataFrame, model) -> Tuple[int,float,str]:
    sig, conf, strat = ctx.signal_manager.evaluate(ctx, df, symbol, model)
    if sig==-1 or conf<CONF_THRESHOLD:
        logger.info(f"[SKIP] {symbol} no/low signal (sig={sig},conf={conf:.2f})")
        return -1,0.0,""
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
        pos=0
    tp = ctx.take_profit_targets.get(symbol)
    if pos and tp and ((pos>0 and price>=tp) or (pos<0 and price<=tp)):
        return True, int(abs(pos)*SCALING_FACTOR), "take_profit"
    action = update_trailing_stop(ctx, symbol, price, pos, atr)
    if action=="exit_long" and pos>0:
        return True, pos, "trailing_stop"
    if action=="exit_short" and pos<0:
        return True, abs(pos), "trailing_stop"
    return False,0,""

def trade_logic(ctx: BotContext, symbol: str, balance: float, model) -> None:
    if not should_enter(ctx, symbol, balance):
        return
    raw = ctx.data_fetcher.get_minute_df(ctx, symbol)
    if raw is None or raw.empty:
        logger.info(f"[SKIP] no minute data for {symbol}")
        return
    df = prepare_indicators(raw)
    sig, conf, _ = signal_and_confirm(ctx, symbol, df, model)
    if sig==-1:
        return
    price, atr = df["Close"].iloc[-1], df["atr"].iloc[-1]
    do_exit, qty, _ = should_exit(ctx, symbol, price, atr)
    if do_exit:
        execute_exit(ctx, symbol, qty)
        return
    size = calculate_entry_size(ctx, symbol, price, atr, conf)
    if size>0:
        try:
            pos = int(ctx.api.get_position(symbol).qty)
        except Exception:
            pos=0
        execute_entry(ctx, symbol, size+abs(pos), "buy" if sig==1 else "sell")

def run_all_trades(model) -> None:
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
        with open(EQUITY_FILE,"r") as f:
            last_cash = float(f.read().strip())
    else:
        last_cash = current_cash

    ctx.kelly_fraction = 0.7 if current_cash>last_cash*1.03 else params["KELLY_FRACTION"]
    with open(EQUITY_FILE,"w") as f:
        f.write(str(current_cash))

    tickers = load_tickers(TICKERS_FILE)
    if not tickers:
        logger.error("❌ No tickers loaded; please check tickers.csv")
        return

    max_workers = min(len(tickers),4)
    with ThreadPoolExecutor(max_workers=max_workers) as execr:
        futures = {execr.submit(_safe_trade, ctx, sym, current_cash, model): sym for sym in tickers}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                fut.result()
            except Exception as e:
                logger.exception(f"[Worker] error processing {sym}: {e}")

def _safe_trade(ctx: BotContext, symbol: str, balance: float, model) -> None:
    try:
        trade_logic(ctx, symbol, balance, model)
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

def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # --- normalize index → Date column
    if df.index.name:
        df = df.reset_index().rename(columns={df.index.name: "Date"})
    else:
        df = df.reset_index().rename(columns={"index": "Date"})
    df.sort_values("Date", inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.dropna(subset=["High","Low","Close"], inplace=True)

    # --- core indicators
    df["vwap"]    = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"])
    df["sma_50"]  = ta.sma(df["Close"], length=50)
    df["sma_200"] = ta.sma(df["Close"], length=200)
    df["rsi"]     = ta.rsi(df["Close"], length=14)
    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    df["macd"], df["macds"] = macd["MACD_12_26_9"], macd["MACDs_12_26_9"]
    df["atr"]     = ta.atr(df["High"], df["Low"], df["Close"], length=ATR_LENGTH)

    # --- Ichimoku: always coerce to two Series
    ich = ta.ichimoku(high=df["High"], low=df["Low"], close=df["Close"])
    # if tuple-like unpack; else DataFrame
    if isinstance(ich, tuple):
        conv_raw, base_raw = ich[0], ich[1]
    else:
        conv_raw, base_raw = ich.iloc[:, 0], ich.iloc[:, 1]

    # if those pieces are still DataFrames, pick their first column
    conv = conv_raw.iloc[:,0] if isinstance(conv_raw, pd.DataFrame) else conv_raw
    base = base_raw.iloc[:,0] if isinstance(base_raw, pd.DataFrame) else base_raw

    df["ichimoku_conv"] = conv
    df["ichimoku_base"] = base

    # --- stochastic RSI
    stoch = ta.stochrsi(df["Close"])
    df["stochrsi"] = stoch["STOCHRSIk_14_14_3_3"]

    # --- clean up
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.dropna(subset=[
        "vwap","sma_50","sma_200","rsi",
        "macd","macds","atr",
        "ichimoku_conv","ichimoku_base","stochrsi"
    ], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
    
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
    try:
        app.run(host="0.0.0.0", port=8080)
    except Exception:
        logger.exception("Healthcheck server failed")

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
