import os
import sys
import time
import csv
import portalocker
import re
import numpy as np
setattr(np, "NaN", np.nan)
import joblib
import asyncio
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import schedule
import logging
import multiprocessing
from functools import wraps
from datetime import datetime, date, timedelta, time as dt_time
from bs4 import BeautifulSoup
from alpaca_trade_api.rest import REST, APIError
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

# ─── PATH CONFIGURATION ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def abspath(fname: str) -> str:
    return os.path.join(BASE_DIR, fname)

TICKERS_FILE        = abspath("tickers.csv")
TRADE_LOG_FILE      = abspath("trades.csv")
SIGNAL_WEIGHTS_FILE = abspath("signal_weights.csv")
EQUITY_FILE         = abspath("last_equity.txt")
PEAK_EQUITY_FILE    = abspath("peak_equity.txt")
HALT_FLAG_FILE      = abspath("halt.flag")
MODEL_FILE          = abspath(os.getenv("MODEL_PATH", "trained_model.pkl"))

# ─── CONFIG & LOGGING ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[ logging.StreamHandler(sys.stdout) ]
)
logger = logging.getLogger(__name__)

# === STRATEGY MODE CONFIGURATION =============================================
class BotMode:
    def __init__(self, mode="balanced"):
        self.mode = mode.lower()
        self.params = self.set_parameters()

    def set_parameters(self):
        if self.mode == "conservative":
            return {
                "KELLY_FRACTION": 0.3,
                "CONF_THRESHOLD": 0.8,
                "CONFIRMATION_COUNT": 3,
                "TAKE_PROFIT_FACTOR": 1.2,
                "DAILY_LOSS_LIMIT": 0.05,
                "CAPITAL_CAP": 0.05,
                "TRAILING_FACTOR": 1.5
            }
        elif self.mode == "aggressive":
            return {
                "KELLY_FRACTION": 0.75,
                "CONF_THRESHOLD": 0.6,
                "CONFIRMATION_COUNT": 1,
                "TAKE_PROFIT_FACTOR": 2.2,
                "DAILY_LOSS_LIMIT": 0.1,
                "CAPITAL_CAP": 0.1,
                "TRAILING_FACTOR": 2.0
            }
        else:  # balanced
            return {
                "KELLY_FRACTION": 0.6,
                "CONF_THRESHOLD": 0.65,
                "CONFIRMATION_COUNT": 2,
                "TAKE_PROFIT_FACTOR": 1.8,
                "DAILY_LOSS_LIMIT": 0.07,
                "CAPITAL_CAP": 0.08,
                "TRAILING_FACTOR": 1.8
            }

    def apply(self):
        return self.params
    
BOT_MODE = os.getenv("BOT_MODE", "balanced")
mode = BotMode(BOT_MODE)
params = mode.apply()

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
ALPACA_API_KEY           = os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET_KEY        = os.getenv("APCA_API_SECRET_KEY")
ALPACA_BASE_URL          = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

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

MARKET_OPEN              = dt_time(6, 30)
MARKET_CLOSE             = dt_time(23, 0)
VOLUME_THRESHOLD         = 500_000
ENTRY_START_OFFSET       = timedelta(minutes=15)
ENTRY_END_OFFSET         = timedelta(minutes=30)

REGIME_LOOKBACK          = 20
REGIME_ATR_THRESHOLD     = 3.0

MODEL_PATH               = MODEL_FILE
RF_ESTIMATORS            = 225
RF_MAX_DEPTH             = 5
ATR_LENGTH               = 12
CONF_THRESHOLD           = params["CONF_THRESHOLD"]
CONFIRMATION_COUNT       = params["CONFIRMATION_COUNT"]
KELLY_FRACTION           = params["KELLY_FRACTION"]
CAPITAL_CAP              = params["CAPITAL_CAP"]

HALT_FLAG_PATH           = HALT_FLAG_FILE

# ─── GLOBAL STATE ─────────────────────────────────────────────────────────────
api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)
confirmation_count       = {}
trailing_extremes        = {}
take_profit_targets      = {}
daily_loss               = 0.0
day_start_equity         = None
data_cache               = {}
api_semaphore            = asyncio.Semaphore(4)

# ─── META-LEARNING SIGNAL DROPPER ───
def load_global_signal_performance(min_trades=10, threshold=0.4):
    path = TRADE_LOG_FILE
    if not os.path.exists(path):
        logger.info("[MetaLearn] no history - allowing all signals")
        return None

    df = pd.read_csv(path)
    df = df.dropna(subset=["exit_price", "entry_price", "signal_tags"])
    df["pnl"] = (df["exit_price"] - df["entry_price"]) * df.apply(
        lambda row: 1 if row["side"] == "buy" else -1, axis=1
    )

    results = {}
    for _, row in df.iterrows():
        tags = row["signal_tags"].split("+")
        for tag in tags:
            results.setdefault(tag, []).append(row["pnl"])

    win_rates = { tag: round(np.mean([1 if p>0 else 0 for p in pnls]),3)
                  for tag,pnls in results.items() if len(pnls)>=min_trades }
    filtered = {tag:wr for tag,wr in win_rates.items() if wr>=threshold}
    logger.info(f"[MetaLearn] Keeping signals: {list(filtered.keys()) or 'none'}")
    return filtered

# ─── TRADE LOGGER ─────────────────────────────────────────────────────────────#
class TradeLogger:
    def __init__(self, path=TRADE_LOG_FILE):
        self.path = path
        if not os.path.exists(self.path):
            with portalocker.Lock(self.path, 'w', timeout=1) as f:
                csv.writer(f).writerow([
                    "symbol","entry_time","entry_price",
                    "exit_time","exit_price","qty","side",
                    "strategy","classification","signal_tags"
                ])

    def log_entry(self, symbol, price, qty, side, strategy, signal_tags=None):
        now = datetime.utcnow().isoformat()
        tag_str = signal_tags or ""
        with portalocker.Lock(self.path, 'a', timeout=1) as f:
            csv.writer(f).writerow([symbol, now, price, "","", qty, side, strategy, "", tag_str])

    def log_exit(self, symbol, exit_price):
        with portalocker.Lock(self.path, 'r+', timeout=1) as f:
            rows = list(csv.reader(f))
            header, data = rows[0], rows[1:]
            for row in data:
                if row[0]==symbol and row[3]=="":
                    entry_time = datetime.fromisoformat(row[1])
                    exit_time  = datetime.utcnow()
                    days = (exit_time-entry_time).days
                    cls  = "day_trade" if days==0 else "swing_trade" if days<5 else "long_trade"
                    row[3],row[4],row[8] = exit_time.isoformat(), exit_price, cls
                    break
            f.seek(0); f.truncate()
            writer=csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)

trade_logger = TradeLogger(path=TRADE_LOG_FILE)

# ─── UTILITY FUNCTIONS ────────────────────────────────────────────────────────
def load_tickers(path):
    tickers=[]
    with open(path,newline="") as f:
        reader=csv.reader(f); next(reader)
        for row in reader:
            t=row[0].strip().upper()
            if t and t not in tickers:
                tickers.append(t)
    return tickers

def within_market_hours():
    now = datetime.now().time()
    start = (datetime.combine(date.today(), MARKET_OPEN) + ENTRY_START_OFFSET).time()
    end   = (datetime.combine(date.today(), MARKET_CLOSE) - ENTRY_END_OFFSET).time()
    return start <= now <= end

def check_daily_loss():
    global daily_loss, day_start_equity
    try:
        acct = api.get_account()
    except:
        return False
    cash = float(acct.cash)
    today = date.today()
    if day_start_equity is None or day_start_equity[0] != today:
        day_start_equity = (today, cash)
        daily_loss = 0.0
        return False
    loss = (day_start_equity[1] - cash) / day_start_equity[1]
    return loss >= DAILY_LOSS_LIMIT

def too_many_positions():
    try:
        return len(api.list_positions()) >= MAX_PORTFOLIO_POSITIONS
    except:
        return False

def check_halt_flag():
    return os.path.exists(HALT_FLAG_PATH)

def check_market_regime():
    df = fetch_data("SPY", period=f"{REGIME_LOOKBACK}d", interval="1d")
    atr_val = ta.atr(df['High'], df['Low'], df['Close'], length=REGIME_LOOKBACK).iloc[-1]
    vol = df['Close'].pct_change().std()
    return (atr_val <= REGIME_ATR_THRESHOLD) or (vol <= 0.015)

def too_correlated(sym):
    if not os.path.exists(TRADE_LOG_FILE):
        return False
    df_tr = pd.read_csv(TRADE_LOG_FILE)
    open_syms = list(df_tr[df_tr.exit_time == ""]['symbol'].unique()) + [sym]
    returns = {}
    for s in open_syms:
        d = fetch_data(s, period='3mo', interval='1d')
        returns[s] = d['Close'].pct_change().dropna()
    min_len = min(len(v) for v in returns.values())
    mat = pd.DataFrame({s: returns[s].tail(min_len) for s in open_syms})
    avg_corr = mat.corr().where(~np.eye(len(open_syms), dtype=bool)).abs().values.mean()
    return avg_corr > CORRELATION_THRESHOLD

def retry(times: int = 3, delay: float = 1.0):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for i in range(times):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"[retry] {fn.__name__} failed ({e}), retry {i+1}/{times}")
                    time.sleep(delay * (2 ** i))
            logger.error(f"[retry] {fn.__name__} failed after {times} attempts.")
            return None
        return wrapper
    return deco

@retry(times=3, delay=0.5)
def fetch_data(ticker, period='3y', interval='1d'):
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    df.reset_index(inplace=True)
    df.ffill().bfill(inplace=True)
    return df

def get_sec_headlines(ticker):
    url = f'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=8-K&count=5'
    headers = {'User-Agent':'AI Trading Bot'}
    try:
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.content, 'lxml')
        texts = []
        for a in soup.find_all('a'):
            if '8-K' in a.text:
                tr = a.find_parent('tr')
                if tr:
                    tds = tr.find_all('td')
                    if len(tds) > 3:
                        texts.append(tds[-1].get_text(strip=True))
        return ' '.join(texts)
    except:
        return ""

def fetch_sentiment(ticker):
    try:
        url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&language=en&pageSize=5&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        if response.status_code != 200:
            return 0.0

        headlines = [a['title'] + " " + a.get('description', '') for a in response.json().get("articles", [])]
        text = ' '.join(headlines).lower()

        score = 0
        if 'beat' in text or 'strong' in text or 'record' in text:
            score += 0.3
        if 'miss' in text or 'weak' in text or 'recall' in text:
            score -= 0.3
        if 'upgrade' in text or 'buy rating' in text:
            score += 0.2
        if 'downgrade' in text or 'sell rating' in text:
            score -= 0.2

        return max(min(score, 0.5), -0.5)
    except:
        return 0.0

# ─── SIGNAL MANAGER ───────────────────────────────────────────────────────────
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
            weight = abs(val) * 10
            return signal, min(weight, 1.0), 'momentum'
        except:
            return -1, 0.0, 'momentum'

    def signal_mean_reversion(self, df):
        try:
            df['zscore'] = (df['Close'] - df['Close'].rolling(self.mean_rev_lookback).mean()) / df['Close'].rolling(self.mean_rev_lookback).std()
            val = df['zscore'].iloc[-1]
            signal = 0 if val > self.mean_rev_zscore_threshold else 1 if val < -self.mean_rev_zscore_threshold else -1
            weight = min(abs(val) / 3, 1.0)
            return signal, weight, 'mean_reversion'
        except:
            return -1, 0.0, 'mean_reversion'

    def signal_ml(self, df, model):
        try:
            feats = ['rsi','sma_50','sma_200','macd','macds','atr']
            X = df[feats].dropna()
            if X.empty: return -1, 0.0, 'ml'
            probs = model.predict_proba(X)
            preds = [int(np.argmax(p)) for p in probs]
            confs = [max(p) for p in probs]
            return preds[-1], confs[-1], 'ml'
        except:
            return -1, 0.0, 'ml'

    def signal_sentiment(self, ticker):
        try:
            val = fetch_sentiment(ticker)
            signal = 1 if val > 0 else 0 if val < 0 else -1
            return signal, min(abs(val), 0.2), 'sentiment'
        except:
            return -1, 0.0, 'sentiment'

    def signal_regime(self):
        try:
            df = fetch_data("SPY", period=f"{REGIME_LOOKBACK}d", interval="1d")
            atr_val = ta.atr(df['High'], df['Low'], df['Close'], length=REGIME_LOOKBACK).iloc[-1]
            signal = 1 if atr_val < self.regime_volatility_threshold else 0
            return signal, 0.6, 'regime'
        except:
            return -1, 0.0, 'regime'

    def signal_stochrsi(self, df):
        try:
            val = df['stochrsi'].iloc[-1]
            signal = 1 if val < 0.2 else 0 if val > 0.8 else -1
            return signal, 0.3, 'stochrsi'
        except:
            return -1, 0.0, 'stochrsi'

    def signal_obv(self, df):
        try:
            obv = ta.obv(df['Close'], df['Volume'])
            df['obv'] = obv
            slope = np.polyfit(range(5), obv.tail(5), 1)[0]
            signal = 1 if slope > 0 else 0 if slope < 0 else -1
            weight = min(abs(slope) / 1e6, 1.0)
            return signal, weight, 'obv'
        except:
            return -1, 0.0, 'obv'

    def signal_vsa(self, df):
        try:
            df['body'] = abs(df['Close'] - df['Open'])
            df['vsa'] = df['Volume'] * df['body']
            score = df['vsa'].iloc[-1]
            signal = 1 if df['Close'].iloc[-1] > df['Open'].iloc[-1] else 0 if df['Close'].iloc[-1] < df['Open'].iloc[-1] else -1
            avg_score = df['vsa'].rolling(20).mean().iloc[-1]
            weight = min(score / avg_score, 1.0)
            return signal, weight, 'vsa'
        except:
            return -1, 0.0, 'vsa'

    def load_signal_weights(self):
        path = SIGNAL_WEIGHTS_FILE
        if not os.path.exists(path):
            return {}
        df = pd.read_csv(path)
        return {row['signal']: row['weight'] for _, row in df.iterrows()}

    def evaluate(self, df, ticker, model):
        signals = []
        allowed_signals = load_global_signal_performance()
        signal_fns = [
            self.signal_momentum,
            self.signal_mean_reversion,
            lambda d: self.signal_ml(d, model),
            lambda d: self.signal_sentiment(ticker),
            lambda d: self.signal_regime(),
            self.signal_stochrsi,
            self.signal_obv,
            self.signal_vsa
        ]
        weight_map = self.load_signal_weights()

        for fn in signal_fns:
            try:
                s, w, label = fn(df)
                # if we have a filter AND this label isn't in it, skip
                if allowed_signals is not None and label not in allowed_signals:
                    continue  # drop underperformers
                if s in [0, 1]:
                    adjusted_weight = weight_map.get(label, w)
                    signals.append((s, adjusted_weight, label))
            except Exception:
                continue

        if not signals:
            return -1, 0.0, 'no_signal'

        score = sum((1 if s == 1 else -1) * w for s, w, _ in signals)
        conf = min(abs(score), 1.0)
        final_sig = 1 if score > 0.5 else 0 if score < -0.5 else -1
        label = '+'.join(lbl for _, _, lbl in signals)

        logger.info(f"[SignalManager] {ticker} | final={final_sig} score={score:.2f} | components: {signals}")
        return final_sig, conf, label

# ─── ORDER EXECUTION & POSITION SIZING ────────────────────────────────────────

def fractional_kelly_size(balance, price, atr, win_prob, payoff_ratio=1.5, base_frac=KELLY_FRACTION):
    peak_file = PEAK_EQUITY_FILE

    # ensure the file exists (with initial value = current balance)
    if not os.path.exists(peak_file):
        with portalocker.Lock(peak_file, 'w', timeout=1) as f:
            f.write(str(balance))
        peak_equity = balance
    else:
        # read & optionally update peak under the same lock
        with portalocker.Lock(peak_file, 'r+', timeout=1) as f:
            content = f.read().strip()
            peak_equity = float(content) if content else balance

            if balance > peak_equity:
                peak_equity = balance
                f.seek(0)
                f.truncate()
                f.write(str(peak_equity))

    drawdown = (peak_equity - balance) / peak_equity

    # Adjust Kelly fraction based on drawdown
    if drawdown > 0.10:
        frac = 0.3
    elif drawdown > 0.05:
        frac = 0.45
    else:
        frac = base_frac

    # standard Kelly
    edge = win_prob - (1 - win_prob) / payoff_ratio
    kelly = max(edge / payoff_ratio, 0) * frac
    dr = kelly * balance
    if atr <= 0:
        return 1

    raw = dr / atr
    cap = (balance * CAPITAL_CAP) / price
    return max(int(min(raw, cap, MAX_POSITION_SIZE)), 1)


def submit_order(ticker, qty, side, order_type_override=None):
    order_type = order_type_override or ORDER_TYPE

    for attempt in range(3):
        try:
            expected_price = None
            if order_type == 'limit':
                quote = api.get_latest_quote(ticker)
                bid, ask = quote.bidprice, quote.askprice
                spread = (ask - bid) if ask and bid else 0
                limit_price = (bid + 0.25 * spread) if side == 'buy' else (ask - 0.25 * spread)
                expected_price = round(limit_price, 2)
                api.submit_order(
                    symbol=ticker, qty=qty, side=side,
                    type='limit', limit_price=expected_price,
                    time_in_force='gtc'
                )
            else:
                quote = api.get_latest_quote(ticker)
                expected_price = quote.askprice if side == 'buy' else quote.bidprice
                api.submit_order(
                    symbol=ticker, qty=qty, side=side,
                    type='market', time_in_force='gtc'
                )

            logger.info(f"[SLIPPAGE] {ticker} expected={expected_price} side={side} qty={qty}")
            return True

        except APIError as e:
            # only catch Alpaca api errors for retryable conditions
            m = re.search(r"requested: (\d+), available: (\d+)", str(e))
            if m and int(m.group(2)) > 0:
                qq = int(m.group(2))
                api.submit_order(symbol=ticker, qty=qq, side=side,
                                 type=order_type, time_in_force='gtc')
                return True
            logger.warning(f"[submit_order] Alpaca APIError on attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)

        except Exception:
            # truly unexpected—log full stack and bail
            logger.exception("Unexpected error in submit_order")
            return False

    logger.error(f"[submit_order] Failed after 3 attempts for {ticker}")
    return False

def update_trailing_stop(ticker, price, qty, atr, factor=TRAILING_FACTOR):
    if qty > 0:
        trailing_extremes[ticker] = max(trailing_extremes.get(ticker, price), price)
        if price < trailing_extremes[ticker] - factor * atr:
            return "exit_long"
    elif qty < 0:
        trailing_extremes[ticker] = min(trailing_extremes.get(ticker, price), price)
        if price > trailing_extremes[ticker] + factor * atr:
            return "exit_short"
    return "hold"

# ─── TRADE LOGIC ─────────────────────────────────────────────────────────────
signal_manager = SignalManager()

def trade_logic(sym, balance, model):
    logger.info(f"[TRADE LOGIC] {sym} – balance={balance:.2f}")

    logger.debug(f"[{sym}] halt={check_halt_flag()}, hours={within_market_hours()}, loss={check_daily_loss()}")
    logger.debug(f"[{sym}] regime={check_market_regime()}, pos_count={len(api.list_positions())}, corr={too_correlated(sym)}")


    if check_halt_flag() or not within_market_hours() or check_daily_loss():
        return
    if not check_market_regime() or too_many_positions() or too_correlated(sym):
        return

    df = fetch_data(sym)
    if df.empty or df['Volume'].tail(20).mean() < VOLUME_THRESHOLD:
        return

    now = datetime.now()
    start = (datetime.combine(now.date(), MARKET_OPEN) + ENTRY_START_OFFSET).time()
    end = (datetime.combine(now.date(), MARKET_CLOSE) - ENTRY_END_OFFSET).time()
    if now.time() < start or now.time() > end:
        return

    # ─── Indicators ───────────────────────────────────────────────────────────
    df['sma_50'] = ta.sma(df['Close'], length=50)
    df['sma_200'] = ta.sma(df['Close'], length=200)
    df['rsi'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['macd'], df['macds'] = macd['MACD_12_26_9'], macd['MACDs_12_26_9']
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=ATR_LENGTH)

    df['vwap'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    ichimoku = ta.ichimoku(df['High'], df['Low'], df['Close'])
    df['ichimoku_base'] = ichimoku['ISA_9']
    df['ichimoku_conv'] = ichimoku['ISB_26']
    df['stochrsi'] = ta.stochrsi(df['Close'])['STOCHRSIk_14_14_3_3']
    df.ffill().bfill(inplace=True)
    df.dropna(subset=[
        'sma_50','sma_200','rsi','macd','macds','atr',
        'vwap','ichimoku_base','ichimoku_conv','stochrsi'
    ], inplace=True)

    sig, conf, strat = signal_manager.evaluate(df, sym, model)
    if sig == -1 or conf < CONF_THRESHOLD:
        return

    price = df['Close'].iloc[-1]
    atr = df['atr'].iloc[-1]
    try:
        pos = int(api.get_position(sym).qty)
    except:
        pos = 0

    # ─── Entry Filters ─────────────────────────────────────────────────────────
    if sig == 1 and price < df['vwap'].iloc[-1]: return
    if sig == 0 and price > df['vwap'].iloc[-1]: return
    if sig == 1 and price < df['ichimoku_base'].iloc[-1]: return
    if sig == 0 and price > df['ichimoku_base'].iloc[-1]: return

    # ─── Take-Profit Scaling ───────────────────────────────────────────────────
    tp = take_profit_targets.get(sym)
    if pos != 0 and tp and ((pos > 0 and price >= tp) or (pos < 0 and price <= tp)):
        exit_qty = int(abs(pos) * SCALING_FACTOR)
        side = 'sell' if pos > 0 else 'buy'
        if submit_order(sym, exit_qty, side):
            trade_logger.log_exit(sym, price)
        trailing_extremes[sym] = price
        return

    prev = confirmation_count.get(sym, {'action': None, 'count': 0})
    if prev['action'] != sig:
        confirmation_count[sym] = {'action': sig, 'count': 1}
        return
    confirmation_count[sym]['count'] += 1
    if confirmation_count[sym]['count'] < CONFIRMATION_COUNT:
        return

    qty = fractional_kelly_size(balance, price, atr, conf)

    if sig == 1 and pos <= 0:
        if submit_order(sym, qty + abs(pos), 'buy'):
            trade_logger.log_entry(sym, price, qty, 'buy', strat, signal_tags=strat)
            take_profit_targets[sym] = price + TAKE_PROFIT_FACTOR * atr
    elif sig == 0 and pos >= 0:
        if submit_order(sym, qty + abs(pos), 'sell'):
            trade_logger.log_entry(sym, price, qty, 'sell', strat, signal_tags=strat)
            take_profit_targets[sym] = price - TAKE_PROFIT_FACTOR * atr

    if pos != 0:
        factor = SECONDARY_TRAIL_FACTOR if tp else TRAILING_FACTOR
        action = update_trailing_stop(sym, price, pos, atr, factor)
        if action == 'exit_long' and pos > 0:
            if submit_order(sym, pos, 'sell'):
                trade_logger.log_exit(sym, price)
        elif action == 'exit_short' and pos < 0:
            if submit_order(sym, abs(pos), 'buy'):
                trade_logger.log_exit(sym, price)

# ─── RUNNER & SCHEDULER ──────────────────────────────────────────────────────
def run_all_trades(model):
    if check_halt_flag():
        logger.info("Trading halted via HALT_FLAG_FILE.")
        return
    try:
        acct = api.get_account()
        current_cash = float(acct.cash)
    except Exception:
        logger.error("Failed to retrieve account balance.")
        return

    # Load last balance from file
    if os.path.exists(EQUITY_FILE):
        with open(EQUITY_FILE, "r") as f:
            last_cash = float(f.read().strip())
    else:
        last_cash = current_cash

    # Detect cash inflow and adjust Kelly if needed
    global KELLY_FRACTION
    if current_cash > last_cash * 1.03:
        logger.info(f"[REWEIGHT] Detected cash inflow. Increasing Kelly fraction from {KELLY_FRACTION:.2f} to 0.7")
        KELLY_FRACTION = 0.7
    else:
        KELLY_FRACTION = 0.6

    # Run trades as normal
    with open (EQUITY_FILE, "w") as f:
        f.write(str(current_cash))

    tickers = load_tickers(TICKERS_FILE)
    pool_size = min(len(tickers), 4)
    with multiprocessing.Pool(pool_size) as pool:
        for sym in tickers:
            pool.apply_async(trade_logic, (sym, current_cash, model))
        pool.close()
        pool.join()

# ─── LOAD MODEL ──────────────────────────────────────────────────────────────
def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        logger.info(f"Loading trained model from {path}")
        return joblib.load(path)
    logger.info("Training new RandomForest model (fallback)")
    model = RandomForestClassifier(n_estimators=RF_ESTIMATORS, max_depth=RF_MAX_DEPTH)
    X = np.random.randn(100, 6)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)
    joblib.dump(model, path)
    return model

# ─── META-LEARNING WEIGHT UPDATER ────────────────────────────────────────────
def update_signal_weights():
    TRADE_LOG = TRADE_LOG_FILE
    WEIGHT_LOG = SIGNAL_WEIGHTS_FILE
    ALPHA = 0.2  # EMA smoothing factor

    if not os.path.exists(TRADE_LOG):
        logger.warning("No TRADE_LOG_FILE found for weight update.")
        return

    df = pd.read_csv(TRADE_LOG)
    df = df.dropna(subset=["exit_price", "entry_price", "signal_tags"])
    df["pnl"] = (df["exit_price"] - df["entry_price"]) * df.apply(
        lambda row: 1 if row["side"] == "buy" else -1, axis=1
    )

    signal_stats = {}
    for _, row in df.iterrows():
        tags = row["signal_tags"].split("+")
        for tag in tags:
            if tag not in signal_stats:
                signal_stats[tag] = []
            signal_stats[tag].append(row["pnl"])

    updated_weights = []
    for tag, pnls in signal_stats.items():
        recent_win_rate = np.mean([1 if p > 0 else 0 for p in pnls])
        weight = round(recent_win_rate, 3)
        updated_weights.append((tag, weight))

    if os.path.exists(WEIGHT_LOG):
        old_df = pd.read_csv(WEIGHT_LOG)
        merged = {}
        for tag, new_w in updated_weights:
            old_row = old_df[old_df.signal == tag]
            if not old_row.empty:
                old_w = old_row.weight.values[0]
                merged[tag] = round(ALPHA * new_w + (1 - ALPHA) * old_w, 3)
            else:
                merged[tag] = new_w
    else:
        merged = {tag: weight for tag, weight in updated_weights}

    df_out = pd.DataFrame(list(merged.items()), columns=["signal", "weight"])
    df_out.to_csv(WEIGHT_LOG, index=False)
    logger.info(f"[MetaLearn] Updated {len(df_out)} signal weights.")

from flask import Flask
import threading

app = Flask(__name__)

@app.route("/health")
def health():
    return "OK", 200

def start_healthcheck():
    app.run(host="0.0.0.0", port=8080)

# ─── MAIN LOOP ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing as _mp
    _mp.set_start_method('spawn')

    # Only the parent kicks off the health‐check thread
    if _mp.current_process().name == "MainProcess":
        threading.Thread(target=start_healthcheck, daemon=True).start()

    model = load_model()
    logger.info("🚀 AI Trading Bot is live!")

    # Scheduled tasks
    schedule.every(1).minutes.do(lambda: run_all_trades(model))
    schedule.every(6).hours.do(update_signal_weights)

    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            logger.exception(f"Scheduler loop error: {e}")
        time.sleep(30)