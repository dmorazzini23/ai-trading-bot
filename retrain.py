import csv
import json
import logging
import os
import random
import warnings

import joblib
import numpy as np
import pandas as pd

import config
from metrics_logger import log_metrics
from utils import safe_to_datetime
from logger import logger

config.reload_env()

# Set deterministic random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch

    torch.manual_seed(SEED)
except ImportError:
    pass
from datetime import date, datetime, time, timedelta, timezone

import requests
from lightgbm import LGBMClassifier
from sklearn.model_selection import ParameterSampler, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pandas_ta as ta

try:
    import optuna
except Exception as e:  # pragma: no cover - optional dependency
    logger.warning("Optuna import failed: %s", e)
    optuna = None

import config

NEWS_API_KEY = config.NEWS_API_KEY

if not NEWS_API_KEY:
    logger.warning("NEWS_API_KEY is not set; sentiment features will be zero")
warnings.filterwarnings("ignore", category=FutureWarning)

MINUTES_REQUIRED = 31
MFI_PERIOD = 14

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def abspath(p: str) -> str:
    return os.path.join(BASE_DIR, p)


def get_git_hash() -> str:
    """Return current git commit short hash if available."""
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception as e:
        logger.debug("git hash lookup failed: %s", e)
        return "unknown"


def atomic_joblib_dump(obj, path: str) -> None:
    """Safely write joblib file atomically."""
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


FEATURE_PERF_FILE = abspath("feature_perf.csv")
INACTIVE_FEATURES_FILE = abspath("inactive_features.json")
HYPERPARAM_LOG_FILE = abspath("hyperparam_log.csv")
MODELS_DIR = abspath("models")
os.makedirs(MODELS_DIR, exist_ok=True)

REWARD_LOG_FILE = abspath("reward_log.csv")


def load_reward_by_band(n: int = 200) -> dict:
    """Return average reward by band from recent history."""
    if not os.path.exists(REWARD_LOG_FILE):
        return {}
    try:
        df = pd.read_csv(REWARD_LOG_FILE).tail(n)
    except Exception as e:
        logger.exception("Failed to read reward log: %s", e)
        return {}
    if df.empty or "band" not in df.columns:
        return {}
    return df.groupby("band")["reward"].mean().to_dict()


def fetch_sentiment(symbol: str) -> float:
    """Lightweight sentiment score using NewsAPI headlines."""
    if not NEWS_API_KEY:
        return 0.0
    try:
        url = f"https://newsapi.org/v2/everything?q={symbol}&pageSize=5&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        if not articles:
            return 0.0
        score = sum(1 for a in articles if "positive" in (a.get("title") or "").lower()) / len(articles)
        return float(score)
    except Exception as e:
        logger.exception("Failed to fetch sentiment for %s: %s", symbol, e)
        return 0.0


##############################################################################
# Inline `detect_regime` so we don’t import bot.py at module load time.
# (Copy your real implementation from bot.py.)
def detect_regime(df: pd.DataFrame) -> str:
    """Simple SMA-based regime detection used by bot and predict scripts."""
    if df is None or df.empty or "close" not in df:
        return "chop"
    close = df["close"].astype(float)
    if len(close) < 200:
        return "chop"
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    if sma50.empty or sma200.empty:
        return "chop"
    if pd.isna(sma50.iloc[-1]) or pd.isna(sma200.iloc[-1]):
        return "chop"
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

    rename_map = {}
    variants = {
        "high": ["High", "HIGH", "H", "h"],
        "low": ["Low", "LOW", "L", "l"],
        "close": ["Close", "CLOSE", "C", "c"],
        "open": ["Open", "OPEN", "O", "o"],
        "volume": ["Volume", "VOLUME", "V", "v"],
    }
    for std, cols in variants.items():
        for col in cols:
            if col in df.columns:
                rename_map[col] = std
    if rename_map:
        df = df.rename(columns=rename_map)

    for col in ["high", "low", "close", "volume"]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame in prepare_indicators")
        df[col] = df[col].astype(float)
    if "open" in df.columns:
        df["open"] = df["open"].astype(float)

    if df.index.name:
        df = df.reset_index().rename(columns={df.index.name: "Date"})
    else:
        df = df.reset_index().rename(columns={"index": "Date"})
    if "timestamp" in df.columns and df["Date"].dtype == object:
        idx = safe_to_datetime(df["timestamp"], context="retrain timestamp")
    else:
        idx = safe_to_datetime(df["Date"], context="retrain date")
    if idx.empty:
        raise ValueError("Invalid date values in dataframe")
    df["Date"] = idx
    df = df.sort_values("Date").set_index("Date")

    # Calculate basic TA indicators
    df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"]).astype(float)
    df.dropna(subset=["vwap"], inplace=True)
    df["rsi"] = ta.rsi(df["close"], length=14).astype("float64")
    df.dropna(subset=["rsi"], inplace=True)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14).astype("float64")
    df.dropna(subset=["atr"], inplace=True)

    # ── New advanced indicators ───────────────────────────────────────────
    df["kc_lower"] = np.nan
    df["kc_mid"] = np.nan
    df["kc_upper"] = np.nan
    try:
        kc = ta.kc(df["high"], df["low"], df["close"], length=20)
        df["kc_lower"] = kc.iloc[:, 0].astype(float)
        df["kc_mid"] = kc.iloc[:, 1].astype(float)
        df["kc_upper"] = kc.iloc[:, 2].astype(float)
    except Exception as e:
        logger.exception("KC indicator failed: %s", e)

    df["atr_band_upper"] = np.nan
    df["atr_band_lower"] = np.nan
    df["avg_vol_20"] = np.nan
    df["dow"] = np.nan
    df["atr_band_upper"] = (df["close"] + 1.5 * df["atr"]).astype(float)
    df["atr_band_lower"] = (df["close"] - 1.5 * df["atr"]).astype(float)
    df["avg_vol_20"] = df["volume"].rolling(20).mean().astype(float)
    df["dow"] = df.index.dayofweek.astype(float)

    df["macd"] = np.nan
    df["macds"] = np.nan
    try:
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df["macd"] = macd["MACD_12_26_9"].astype(float)
        df["macds"] = macd["MACDs_12_26_9"].astype(float)
    except Exception as e:
        logger.exception("MACD calculation failed: %s", e)

    # Additional indicators for richer ML features
    df["bb_upper"] = np.nan
    df["bb_lower"] = np.nan
    df["bb_percent"] = np.nan
    try:
        bb = ta.bbands(df["close"], length=20)
        df["bb_upper"] = bb["BBU_20_2.0"].astype(float)
        df["bb_lower"] = bb["BBL_20_2.0"].astype(float)
        df["bb_percent"] = bb["BBP_20_2.0"].astype(float)
    except Exception as e:
        logger.exception("Bollinger Bands failed: %s", e)

    df["adx"] = np.nan
    df["dmp"] = np.nan
    df["dmn"] = np.nan
    try:
        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        df["adx"] = adx["ADX_14"].astype(float)
        df["dmp"] = adx["DMP_14"].astype(float)
        df["dmn"] = adx["DMN_14"].astype(float)
    except Exception as e:
        logger.exception("ADX calculation failed: %s", e)

    df["cci"] = np.nan
    try:
        df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=20).astype(float)
    except Exception as e:
        logger.exception("CCI calculation failed: %s", e)

    try:
        mfi_vals = ta.mfi(
            df["high"],
            df["low"],
            df["close"],
            df["volume"],
            length=MFI_PERIOD,
        ).astype(float)
        df["mfi_14"] = mfi_vals
        df.dropna(subset=["mfi_14"], inplace=True)
    except Exception as e:
        logger.exception("MFI calculation failed: %s", e)
        df["mfi_14"] = np.nan

    df["tema"] = np.nan
    try:
        df["tema"] = ta.tema(df["close"], length=10).astype(float)
    except Exception as e:
        logger.exception("TEMA calculation failed: %s", e)

    df["willr"] = np.nan
    try:
        df["willr"] = ta.willr(df["high"], df["low"], df["close"], length=14).astype(float)
    except Exception as e:
        logger.exception("Williams %R calculation failed: %s", e)

    df["psar_long"] = np.nan
    df["psar_short"] = np.nan
    try:
        psar = ta.psar(df["high"], df["low"], df["close"])
        df["psar_long"] = psar["PSARl_0.02_0.2"].astype(float)
        df["psar_short"] = psar["PSARs_0.02_0.2"].astype(float)
    except Exception as e:
        logger.exception("PSAR calculation failed: %s", e)

    df["ichimoku_conv"] = np.nan
    df["ichimoku_base"] = np.nan
    try:
        ich = ta.ichimoku(high=df["high"], low=df["low"], close=df["close"])
        conv = ich[0] if isinstance(ich, tuple) else ich.iloc[:, 0]
        base = ich[1] if isinstance(ich, tuple) else ich.iloc[:, 1]
        df["ichimoku_conv"] = (conv.iloc[:, 0] if hasattr(conv, "iloc") else conv).astype(float)
        df["ichimoku_base"] = (base.iloc[:, 0] if hasattr(base, "iloc") else base).astype(float)
    except Exception as e:
        logger.exception("Ichimoku calculation failed: %s", e)

    df["stochrsi"] = np.nan
    try:
        st = ta.stochrsi(df["close"])
        df["stochrsi"] = st["STOCHRSIk_14_14_3_3"].astype(float)
    except Exception as e:
        logger.exception("StochRSI calculation failed: %s", e)

    # --- Multi-timeframe fusion ---
    df["ret_5m"] = np.nan
    df["ret_1h"] = np.nan
    df["ret_d"] = np.nan
    df["ret_w"] = np.nan
    df["vol_norm"] = np.nan
    df["5m_vs_1h"] = np.nan
    df["vol_5m"] = np.nan
    df["vol_1h"] = np.nan
    df["vol_d"] = np.nan
    df["vol_w"] = np.nan
    df["vol_ratio"] = np.nan
    df["mom_agg"] = np.nan
    df["lag_close_1"] = np.nan
    df["lag_close_3"] = np.nan
    try:
        df["ret_5m"] = df["close"].pct_change(5).astype(float)
        df["ret_1h"] = df["close"].pct_change(60).astype(float)
        df["ret_d"] = df["close"].pct_change(390).astype(float)
        df["ret_w"] = df["close"].pct_change(1950).astype(float)
        df["vol_norm"] = (df["volume"].rolling(60).mean() / df["volume"].rolling(5).mean()).astype(float)
        df["5m_vs_1h"] = (df["ret_5m"] - df["ret_1h"]).astype(float)
        df["vol_5m"] = df["close"].pct_change().rolling(5).std().astype(float)
        df["vol_1h"] = df["close"].pct_change().rolling(60).std().astype(float)
        df["vol_d"] = df["close"].pct_change().rolling(390).std().astype(float)
        df["vol_w"] = df["close"].pct_change().rolling(1950).std().astype(float)
        df["vol_ratio"] = (df["vol_5m"] / df["vol_1h"]).astype(float)
        df["mom_agg"] = (df["ret_5m"] + df["ret_1h"] + df["ret_d"]).astype(float)
        df["lag_close_1"] = df["close"].shift(1).astype(float)
        df["lag_close_3"] = df["close"].shift(3).astype(float)
    except Exception as e:
        logger.exception("Multi-timeframe features failed: %s", e)

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
        df["sma_50"] = np.nan
        df["sma_200"] = np.nan
        df["sma_50"] = ta.sma(df["close"], length=50).astype(float)
        df["sma_200"] = ta.sma(df["close"], length=200).astype(float)
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
            bars = ctx.data_fetcher.get_minute_df(ctx, sym)
        except Exception as e:
            bars = None
            logger.exception("[gather_minute_data] %s fetch failed: %s", sym, e)
            print(f"[gather_minute_data] ✗ {sym} → exception {start_dt}→{end_dt}: {e}")

        if bars is None or bars.empty:
            print(f"[gather_minute_data] – {sym} → 0 rows")
        else:
            print(f"[gather_minute_data] ✓ {sym} → {len(bars)} rows")
            raw_store[sym] = bars

    if not raw_store:
        logger.critical(
            "No symbols returned valid data for the day. Cannot trade or retrain. Check your data subscription."
        )

    return raw_store


def build_feature_label_df(
    raw_store: dict[str, pd.DataFrame],
    Δ_minutes: int = 30,
    threshold_pct: float = 0.002,
) -> pd.DataFrame:
    """
    Build a combined DataFrame of (feature_vector, label) for each minute-slice.
    Labels are generated using a simulated trade with 0.05% buy slippage and
    0.05% sell slippage. If a symbol has fewer than Δ_minutes+1 rows, it is
    skipped with a notice.
    """
    rows = []
    for sym, raw in raw_store.items():
        try:
            if raw is None or raw.empty:
                print(f"[build_feature_label_df] – {sym} returned no minute data; skipping symbol.")
                continue
            min_required = max(int(MINUTES_REQUIRED * 0.7), 5)
            if raw.shape[0] < min_required:
                print(
                    f"[build_feature_label_df] – skipping {sym}, only {raw.shape[0]} < {min_required} "
                    f"(70% of {MINUTES_REQUIRED})"
                )
                continue

            for col in list(raw.columns):
                if col.lower() in ["high", "low", "close", "volume"]:
                    raw = raw.rename(columns={col: col.lower()})
            if (
                "close" not in raw.columns
                or "high" not in raw.columns
                or "low" not in raw.columns
                or "volume" not in raw.columns
            ):
                print(f"[build_feature_label_df] – skipping {sym}, missing price/volume columns")
                continue

            for col in ["high", "low", "close", "volume"]:
                raw[col] = raw[col].astype(float)

            closes = raw["close"].values

            feat = prepare_indicators(raw, freq="intraday")
            if feat.empty:
                print(f"[build_feature_label_df] – {sym} indicators empty after dropna")
                continue

            sent = fetch_sentiment(sym)
            feat["sentiment"] = sent

            regime_info = detect_regime(raw)
            if isinstance(regime_info, pd.Series):
                regimes = regime_info.reset_index(drop=True)
            else:
                regimes = pd.Series([regime_info] * len(feat))

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
        except KeyError as e:
            logger.exception("[build_feature_label_df] %s missing data: %s", sym, e)
            print(f"[build_feature_label_df] – skipping {sym}, KeyError: {e}")
            continue

    df_all = pd.DataFrame(rows).dropna()
    return df_all


def log_hyperparam_result(regime: str, generation: int, params: dict, score: float) -> None:
    row = [
        datetime.now(timezone.utc).isoformat(),
        regime,
        generation,
        json.dumps(params),
        score,
        SEED,
        get_git_hash(),
    ]
    header = [
        "timestamp",
        "regime",
        "generation",
        "params",
        "score",
        "seed",
        "git_hash",
    ]
    write_header = not os.path.exists(HYPERPARAM_LOG_FILE)
    try:
        with open(HYPERPARAM_LOG_FILE, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow(row)
    except Exception as e:
        logger.exception("Failed to log hyperparameters: %s", e)


def save_model_version(clf, regime: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = f"model_{regime}_{ts}.pkl"
    path = os.path.join(MODELS_DIR, filename)
    try:
        atomic_joblib_dump(clf, path)
    except Exception as e:
        logger.exception("Failed to save model %s: %s", regime, e)
        raise
    log_hyperparam_result(regime, -1, {"model_path": filename}, 0.0)
    log_metrics(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "model_checkpoint",
            "regime": regime,
            "model_path": filename,
            "seed": SEED,
            "git_hash": get_git_hash(),
        }
    )
    link_path = MODEL_FILES.get(regime, os.path.join(MODELS_DIR, f"model_{regime}.pkl"))
    try:
        if os.path.islink(link_path) or os.path.exists(link_path):
            os.remove(link_path)
        os.symlink(filename, link_path)
    except Exception as e:
        logger.exception("Failed to update symlink for %s: %s", regime, e)
    return path


def evolutionary_search(
    X,
    y,
    base_params: dict,
    param_space: dict,
    generations: int = 3,
    population: int = 10,
    elite: int = 3,
    scoring: str = "roc_auc",
) -> dict:
    """Simple evolutionary hyperparameter search."""
    best_params = base_params.copy()
    best_score = -np.inf
    population_params = list(ParameterSampler(param_space, n_iter=population, random_state=SEED))
    for gen in range(generations):
        scores = []
        for params in population_params:
            cfg = base_params.copy()
            cfg.update(params)
            clf = LGBMClassifier(**cfg)
            try:
                cv_scores = cross_val_score(clf, X, y, cv=3, scoring=scoring)
            except Exception as e:
                logger.exception("CV failed for params %s: %s", params, e)
                continue
            if len(cv_scores) == 0:
                continue
            score = float(cv_scores.mean())
            scores.append(score)
        if not scores:
            logger.warning("No successful CV scores in generation %s", gen)
            continue
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


def optuna_search(
    X,
    y,
    base_params: dict,
    param_space: dict,
    n_trials: int = 20,
    scoring: str = "roc_auc",
) -> dict:
    """Hyperparameter tuning using Optuna if available."""
    if optuna is None:
        logger.warning("Optuna not installed; falling back to evolutionary search")
        return evolutionary_search(X, y, base_params, param_space, generations=3, population=n_trials)

    def objective(trial):
        params = {k: trial.suggest_categorical(k, v) for k, v in param_space.items()}
        cfg = base_params.copy()
        cfg.update(params)
        clf = LGBMClassifier(**cfg)
        score = cross_val_score(clf, X, y, cv=3, scoring=scoring).mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return {**base_params, **study.best_params}


def retrain_meta_learner(
    ctx,
    symbols,
    lookback_days: int = 5,
    Δ_minutes: int = 30,
    threshold_pct: float = 0.002,
    force: bool = False,
) -> bool:
    now = datetime.now(timezone.utc)
    if not force:
        if now.weekday() >= 5:
            logger.info(
                "[retrain_meta_learner] Weekend detected; skipping",
                extra={"weekday": now.weekday()},
            )
            return False
        market_open = time(9, 30)
        market_close = time(16, 0)
        if not (market_open <= now.time() <= market_close):
            logger.info(
                "[retrain_meta_learner] Outside market hours; skipping",
                extra={"time": now.time().strftime("%H:%M")},
            )
            return False

    logger.info(f"Starting meta-learner retraining at {now:%Y-%m-%d %H:%M:%S}")
    raw_store = gather_minute_data(ctx, symbols, lookback_days=lookback_days)
    if not raw_store:
        logger.critical(
            "No symbols returned valid data for the day. Cannot trade or retrain. Check your data subscription."
        )
        return False

    filtered: dict[str, pd.DataFrame] = {}
    for sym, df in raw_store.items():
        if df is None or df.empty:
            logger.warning(f"[retrain_meta_learner] {sym} empty minute df; skipping")
            continue
        filtered[sym] = df
    raw_store = filtered
    if not raw_store:
        logger.warning("All minute DataFrames empty; skipping retrain")
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

        if len(y_train) < 3:
            logger.warning(
                "[retrain_meta_learner] Not enough training samples for %s; skipping",
                regime,
            )
            continue

        pos_ratio = y_train.mean()
        scoring = "f1" if 0.4 <= pos_ratio <= 0.6 else "roc_auc"
        print(f"  ✔ Training {regime} model using scoring='{scoring}' (pos_ratio={pos_ratio:.3f})")

        base_params = dict(objective="binary", n_jobs=-1, random_state=SEED)
        search_space = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [-1, 4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.15],
            "num_leaves": [31, 50, 75],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
        best_hyper = optuna_search(
            X_train,
            y_train,
            base_params,
            search_space,
            n_trials=20,
            scoring=scoring,
        )
        clf = LGBMClassifier(**best_hyper)
        pipe = make_pipeline(StandardScaler(), clf)
        if X_train.empty:
            logger.warning(f"[retrain_meta_learner] Training data empty for {regime}; skipping model fit")
            continue
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
            imp_df = pd.DataFrame(
                {
                    "timestamp": [datetime.now(timezone.utc).isoformat()] * len(importances),
                    "feature": importances.index,
                    "importance": importances.values,
                }
            )
            if os.path.exists(FEATURE_PERF_FILE):
                imp_df.to_csv(FEATURE_PERF_FILE, mode="a", header=False, index=False)
            else:
                imp_df.to_csv(FEATURE_PERF_FILE, index=False)
        except Exception as e:
            logger.exception("Failed to log feature importances for %s: %s", regime, e)

        path = save_model_version(pipe, regime)
        print(f"  ✔ Saved {regime} model to {path}")
        log_metrics(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "retrain_model",
                "regime": regime,
                "metric": metric,
                "scoring": scoring,
                "params": json.dumps(best_hyper),
                "seed": SEED,
                "git_hash": get_git_hash(),
            }
        )
        trained_any = True

    try:
        if os.path.exists(FEATURE_PERF_FILE):
            perf = pd.read_csv(FEATURE_PERF_FILE)
            if perf.empty:
                raise ValueError("feature_perf.csv is empty")
            recent = perf.groupby("feature").tail(5)
            means = recent.groupby("feature")["importance"].mean()
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
            with open(INACTIVE_FEATURES_FILE, "w") as f:
                json.dump(sorted(current), f)
    except Exception as e:
        logger.exception("Failed updating inactive features: %s", e)

    try:
        band_rewards = load_reward_by_band()
        if band_rewards:
            print(f"  ✔ Avg rewards by band: {band_rewards}")
    except Exception as e:
        logger.exception("Failed to load reward by band: %s", e)

    return trained_any
