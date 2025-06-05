import os
import json
import csv
import random
import joblib
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv

from datetime import datetime, date, time, timedelta
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, ParameterSampler, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

import pandas_ta as ta

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def abspath(p: str) -> str:
    return os.path.join(BASE_DIR, p)
FEATURE_PERF_FILE = abspath("feature_perf.csv")
INACTIVE_FEATURES_FILE = abspath("inactive_features.json")
HYPERPARAM_LOG_FILE = abspath("hyperparam_log.csv")
MODELS_DIR = abspath("models")
os.makedirs(MODELS_DIR, exist_ok=True)

REWARD_LOG_FILE = abspath("reward_log.csv")

def load_reward_by_band(n: int = 200) -> dict:
    if not os.path.exists(REWARD_LOG_FILE):
        return {}
    df = pd.read_csv(REWARD_LOG_FILE).tail(n)
    if "band" not in df.columns:
        return {}
    return df.groupby("band")["reward"].mean().to_dict()

def fetch_sentiment(symbol: str) -> float:
    """Lightweight sentiment score using NewsAPI headlines."""
    if not NEWS_API_KEY:
        return 0.0
    try:
        url = (
            f"https://newsapi.org/v2/everything?q={symbol}&pageSize=5&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        if not articles:
            return 0.0
        score = sum(1 for a in articles if "positive" in (a.get("title") or "").lower()) / len(articles)
        return float(score)
    except Exception:
        return 0.0

##############################################################################
# Inline `detect_regime` so we don’t import bot.py at module load time.
# (Copy your real implementation from bot.py.)
def detect_regime(df: pd.DataFrame) -> str:
    """Simple SMA-based regime detection used by bot and predict scripts."""
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
##############################################################################

# Output models for each regime
MODEL_FILES = {
    "bull": os.path.join(MODELS_DIR, "model_bull.pkl"),
    "bear": os.path.join(MODELS_DIR, "model_bear.pkl"),
    "chop": os.path.join(MODELS_DIR, "model_chop.pkl"),
}

# ─── COPY&PASTE of prepare_indicators (unchanged) ─────────────────
def prepare_indicators(df: pd.DataFrame, freq: str = "daily") -> pd.DataFrame:
    df = df.copy()
    if df.index.name:
        df = df.reset_index().rename(columns={df.index.name: "Date"})
    else:
        df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    # Calculate basic TA indicators
    df["vwap"] = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"])
    df["rsi"] = ta.rsi(df["Close"], length=14)
    df["atr"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

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
        df["macd"] = macd["MACD_12_26_9"]
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

    try:
        df["mfi"] = ta.mfi(df["High"], df["Low"], df["Close"], df["Volume"], length=14)
    except Exception:
        df["mfi"] = np.nan

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

    required = [
        "vwap", "rsi", "atr", "macd", "macds",
        "ichimoku_conv", "ichimoku_base", "stochrsi"
    ]
    if freq == "daily":
        df["sma_50"] = ta.sma(df["Close"], length=50)
        df["sma_200"] = ta.sma(df["Close"], length=200)
        required += ["sma_50", "sma_200"]
        df.dropna(subset=required, how="any", inplace=True)
    else:  # intraday
        df.dropna(subset=required, how="all", inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df
# ──────────────────────────────────────────────────────────────────────────────

# No top‐level import of bot.py ⇒ avoids metric re‐registration.

def gather_minute_data(ctx, symbols, lookback_days: int = 5) -> dict[str, pd.DataFrame]:
    """
    For each symbol, fetch minute bars from Alpaca day‐by‐day over the last `lookback_days`.
    Prints per‐symbol row‐counts, and only stores those with ≥ 1 row.
    """
    raw_store: dict[str, pd.DataFrame] = {}
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=lookback_days)

    print(f"[gather_minute_data] start={start_dt}, end={end_dt}, symbols={symbols}")
    for sym in symbols:
        bars = None
        try:
            bars = ctx.data_fetcher.get_historical_minute(ctx, sym, start_dt, end_dt)
        except Exception as e:
            bars = None
            print(f"[gather_minute_data] ✗ {sym} → exception {start_dt}→{end_dt}: {e}")

        if bars is None or bars.empty:
            print(f"[gather_minute_data] – {sym} → 0 rows")
        else:
            print(f"[gather_minute_data] ✓ {sym} → {len(bars)} rows")
            raw_store[sym] = bars

    return raw_store

def build_feature_label_df(
    raw_store: dict[str, pd.DataFrame],
    Δ_minutes: int = 30,
    threshold_pct: float = 0.002
) -> pd.DataFrame:
    """
    Build a combined DataFrame of (feature_vector, label) for each minute-slice.
    Labels are generated using a simulated trade with 0.05% buy slippage and
    0.05% sell slippage. If a symbol has fewer than Δ_minutes+1 rows, it is
    skipped with a notice.
    """
    rows = []
    for sym, raw in raw_store.items():
        if raw.shape[0] < (Δ_minutes + 1):
            print(f"[build_feature_label_df] – skipping {sym}, only {raw.shape[0]} < {Δ_minutes + 1}")
            continue

        feat = prepare_indicators(raw, freq="intraday")
        if feat.empty:
            print(f"[build_feature_label_df] – {sym} indicators empty after dropna")
            continue

        sent = fetch_sentiment(sym)
        feat["sentiment"] = sent

        # Determine regime series for this symbol's data
        regime_info = detect_regime(raw)
        if isinstance(regime_info, pd.Series):
            regimes = regime_info.reset_index(drop=True)
        else:
            regimes = pd.Series([regime_info] * len(feat))

        closes = raw["Close"].values
        n = len(feat)
        for i in range(n - Δ_minutes):
            buy_fill = closes[i] * (1 + 0.0005)
            sell_fill = closes[i + Δ_minutes] * (1 - 0.0005)
            ret_pct = (sell_fill / buy_fill) - 1.0
            label = 1 if ret_pct >= threshold_pct else 0

            row = feat.iloc[i].copy().to_dict()
            row["regime"] = regimes.iloc[i] if len(regimes) > i else regimes.iloc[-1]
            row["label"] = label
            rows.append(row)

    df_all = pd.DataFrame(rows).dropna()
    return df_all


def log_hyperparam_result(regime: str, generation: int, params: dict, score: float) -> None:
    row = [datetime.utcnow().isoformat(), regime, generation, json.dumps(params), score]
    header = ["timestamp", "regime", "generation", "params", "score"]
    write_header = not os.path.exists(HYPERPARAM_LOG_FILE)
    with open(HYPERPARAM_LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


def save_model_version(clf, regime: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    filename = f"model_{regime}_{ts}.pkl"
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(clf, path)
    log_hyperparam_result(regime, -1, {"model_path": filename}, 0.0)
    link_path = MODEL_FILES.get(regime, os.path.join(MODELS_DIR, f"model_{regime}.pkl"))
    try:
        if os.path.islink(link_path) or os.path.exists(link_path):
            os.remove(link_path)
        os.symlink(filename, link_path)
    except Exception:
        pass
    return path


def evolutionary_search(X, y, base_params: dict, param_space: dict, generations: int = 3, population: int = 10, elite: int = 3, scoring: str = "roc_auc") -> dict:
    """Simple evolutionary hyperparameter search."""
    best_params = base_params.copy()
    best_score = -np.inf
    population_params = list(ParameterSampler(param_space, n_iter=population, random_state=42))
    for gen in range(generations):
        scores = []
        for params in population_params:
            cfg = base_params.copy()
            cfg.update(params)
            clf = LGBMClassifier(**cfg)
            cv_scores = cross_val_score(clf, X, y, cv=3, scoring=scoring)
            score = float(cv_scores.mean())
            scores.append(score)
        ranked = sorted(zip(scores, population_params), key=lambda t: t[0], reverse=True)
        for rank, (scr, pr) in enumerate(ranked):
            log_hyperparam_result("", gen, pr, scr)
        if ranked[0][0] > best_score:
            best_score = ranked[0][0]
            best_params = {**base_params, **ranked[0][1]}
        # next generation mutate top elite
        new_pop = [ranked[i][1].copy() for i in range(min(elite, len(ranked)))]
        while len(new_pop) < population:
            parent = random.choice(new_pop).copy()
            for k, values in param_space.items():
                if random.random() < 0.3:
                    parent[k] = random.choice(values)
            new_pop.append(parent)
        population_params = new_pop
    return best_params

def retrain_meta_learner(
    ctx,
    symbols,
    lookback_days: int = 5,
    Δ_minutes: int = 30,
    threshold_pct: float = 0.002,
    force: bool = False,
) -> bool:
    now = datetime.now()
    if not force:
        if now.weekday() >= 5:
            print(f"[retrain_meta_learner] Weekend detected (weekday={now.weekday()}). Skipping retrain.")
            return False
        market_open = time(9, 30)
        market_close = time(16, 0)
        if not (market_open <= now.time() <= market_close):
            print(f"[retrain_meta_learner] Outside market hours ({now.time().strftime('%H:%M')}). Skipping retrain.")
            return False

    print(f"[{now:%Y-%m-%d %H:%M:%S}] ▶ Starting meta‐learner retraining…")
    raw_store = gather_minute_data(ctx, symbols, lookback_days=lookback_days)
    if not raw_store:
        print("  ⚠️ No symbol returned any minute bars → skipping retrain.")
        return False

    df_all = build_feature_label_df(raw_store, Δ_minutes=Δ_minutes, threshold_pct=threshold_pct)
    if df_all.empty:
        print("  ⚠️ No usable rows after building (Δ, threshold) → skipping retrain.")
        return False

    trained_any = False
    for regime, subset in df_all.groupby("regime"):
        if subset.empty or regime not in MODEL_FILES:
            continue

        X = subset.drop(columns=["label", "regime"])
        y = subset["label"]

        split_idx = int(len(subset) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        pos_ratio = y_train.mean()
        scoring = "f1" if 0.4 <= pos_ratio <= 0.6 else "roc_auc"
        print(f"  ✔ Training {regime} model using scoring='{scoring}' (pos_ratio={pos_ratio:.3f})")

        base_params = dict(objective="binary", n_jobs=-1, random_state=42)
        search_space = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [-1, 4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.15],
            "num_leaves": [31, 50, 75],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
        best_hyper = evolutionary_search(X_train, y_train, base_params, search_space, generations=3, population=8, scoring=scoring)
        clf = LGBMClassifier(**best_hyper)
        pipe = make_pipeline(StandardScaler(), clf)
        pipe.fit(X_train, y_train)
        print(f"  ✔ {regime} best params: {best_hyper}")

        from sklearn.metrics import f1_score, roc_auc_score
        if scoring == "f1":
            val_pred = pipe.predict(X_val)
            metric = f1_score(y_val, val_pred)
            print(f"  ✔ {regime} holdout F1 = {metric:.4f}")
        else:
            val_probs = pipe.predict_proba(X_val)[:, 1]
            metric = roc_auc_score(y_val, val_probs)
            print(f"  ✔ {regime} holdout AUC = {metric:.4f}")

        try:
            importances = pd.Series(
                pipe.named_steps["lgbmclassifier"].feature_importances_,
                index=X.columns,
            )
            print("  ✔ Top feature importances:")
            print(importances.sort_values(ascending=False).head(10))
            imp_df = pd.DataFrame({
                'timestamp': [datetime.utcnow().isoformat()] * len(importances),
                'feature': importances.index,
                'importance': importances.values
            })
            if os.path.exists(FEATURE_PERF_FILE):
                imp_df.to_csv(FEATURE_PERF_FILE, mode='a', header=False, index=False)
            else:
                imp_df.to_csv(FEATURE_PERF_FILE, index=False)
        except Exception:
            pass

        path = save_model_version(pipe, regime)
        print(f"  ✔ Saved {regime} model to {path}")
        trained_any = True

    try:
        if os.path.exists(FEATURE_PERF_FILE):
            perf = pd.read_csv(FEATURE_PERF_FILE)
            recent = perf.groupby('feature').tail(5)
            means = recent.groupby('feature')['importance'].mean()
            threshold = means.quantile(0.1)
            inactive = means[means < threshold].index.tolist()
            if os.path.exists(INACTIVE_FEATURES_FILE):
                with open(INACTIVE_FEATURES_FILE) as f:
                    current = set(json.load(f))
            else:
                current = set()
            current.update(inactive)
            revived = means[means >= threshold].index.tolist()
            current.difference_update(revived)
            with open(INACTIVE_FEATURES_FILE, 'w') as f:
                json.dump(sorted(current), f)
    except Exception:
        pass

    try:
        band_rewards = load_reward_by_band()
        if band_rewards:
            print(f"  ✔ Avg rewards by band: {band_rewards}")
    except Exception:
        pass

    return trained_any
