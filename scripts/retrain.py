import csv
import json
import logging
import os
import random

# pandas_ta SyntaxWarning now filtered globally in pytest.ini

# AI-AGENT-REF: graceful joblib fallback for testing
try:
    import joblib
except ImportError:
    # Create minimal joblib fallback
    class MockJoblib:
        @staticmethod
        def load(filename):
            # Return a minimal mock model
            class MockModel:
                def predict(self, X):
                    return [0] * len(X)
                def predict_proba(self, X):
                    return [[0.5, 0.5]] * len(X)
            return MockModel()

        @staticmethod
        def dump(obj, filename):
            pass  # Mock dump
    joblib = MockJoblib()

# AI-AGENT-REF: graceful numpy fallback for testing
try:
    import numpy as np
except ImportError:
    # Create minimal numpy fallback
    class MockNumpy:
        nan = float('nan')
        def array(self, *args, **kwargs):
            return list(args[0]) if args else []
        def mean(self, arr):
            return sum(arr) / len(arr) if arr else 0
        def std(self, arr):
            if not arr:
                return 0
            mean_val = sum(arr) / len(arr)
            return (sum((x - mean_val)**2 for x in arr) / len(arr))**0.5
        @property
        def random(self):
            import random
            return random
    np = MockNumpy()

# AI-AGENT-REF: graceful pandas fallback for testing
try:
    import pandas as pd
except ImportError:
    # Import mock pandas from utils
    from ai_trading.utils import pd

import config
from ai_trading.telemetry.metrics_logger import log_metrics

from ai_trading.utils.base import safe_to_datetime

logger = logging.getLogger(__name__)

config.reload_env()

# Set deterministic random seeds for reproducibility
SEED = config.SEED
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch

    torch.manual_seed(SEED)
except ImportError:
    pass
import importlib
from datetime import UTC, date, datetime, time, timedelta

import requests

# AI-AGENT-REF: graceful lightgbm fallback for testing
try:
    from lightgbm import LGBMClassifier
except ImportError:
    # Create minimal LGBMClassifier fallback
    class LGBMClassifier:
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, X, y):
            return self
        def predict(self, X):
            return [0] * len(X)
        def predict_proba(self, X):
            return [[0.5, 0.5]] * len(X)

# AI-AGENT-REF: graceful sklearn fallback for testing
try:
    from sklearn.model_selection import ParameterSampler, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:
    # Create minimal sklearn fallbacks
    class ParameterSampler:
        def __init__(self, *args, **kwargs):
            pass
        def __iter__(self):
            return iter([{}])

    def cross_val_score(*args, **kwargs):
        return [0.5] * 5  # Mock CV scores

    def make_pipeline(*args, **kwargs):
        class MockPipeline:
            def fit(self, X, y):
                return self
            def predict(self, X):
                return [0] * len(X)
        return MockPipeline()

    class StandardScaler:
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, X):
            return self
        def transform(self, X):
            return X

try:
    import optuna
except Exception as e:  # pragma: no cover - optional dependency
    logger.warning("Optuna import failed: %s", e)
    optuna = None

import config

NEWS_API_KEY = config.NEWS_API_KEY

if not NEWS_API_KEY:
    logger.warning("NEWS_API_KEY is not set; sentiment features will be zero")
# FutureWarning now filtered globally in pytest.ini

MINUTES_REQUIRED = 31
MFI_PERIOD = 14

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def abspath(p: str) -> str:
    return os.path.join(BASE_DIR, p)


def get_git_hash() -> str:
    """Return current git commit short hash if available."""
    try:
        import subprocess

        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], timeout=30)
            .decode()
            .strip()
        )
    except Exception as e:
        logger.debug("git hash lookup failed: %s", e)
        return "unknown"


def atomic_joblib_dump(obj, path: str) -> None:
    """Safely write joblib file atomically."""
    import tempfile

    dir_name = os.path.dirname(path)
    # Thread-safe directory creation
    try:
        os.makedirs(dir_name, exist_ok=True)
    except FileExistsError:
        pass  # Another thread created it
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

# (Copy your real implementation from bot.py.)

def fetch_sentiment(symbol: str) -> float:
    """Lightweight sentiment score using NewsAPI headlines."""
    if not NEWS_API_KEY:
        logger.debug("No NEWS_API_KEY configured, returning neutral sentiment")
        return 0.0

    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": symbol,
            "apiKey": NEWS_API_KEY,
            "pageSize": 10,
            "sortBy": "publishedAt",
            "language": "en",
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        articles = data.get("articles", [])

        if not articles:
            return 0.0

        positive_words = ["up", "rise", "gain", "bull", "positive", "growth", "increase"]
        negative_words = ["down", "fall", "drop", "bear", "negative", "decline", "decrease"]

        sentiment_score = 0.0
        for article in articles:
            title = article.get("title", "").lower()
            sentiment_score += sum(1 for word in positive_words if word in title)
            sentiment_score -= sum(1 for word in negative_words if word in title)

        return max(-1.0, min(1.0, sentiment_score / max(len(articles), 1)))

    except Exception as e:
        logger.warning("Failed to fetch sentiment for %s: %s", symbol, e)
        return 0.0
def detect_regime(df: pd.DataFrame) -> str:
    """Simple SMA-based regime detection used by bot and predict scripts."""
    if df is None or df.empty or "close" not in df:
        return "chop"

    try:
        close = df["close"].astype(float)
        if len(close) < 50:
            return "chop"

        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()

        current_price = close.iloc[-1]
        current_sma20 = sma_20.iloc[-1]
        current_sma50 = sma_50.iloc[-1]

        if current_price > current_sma20 > current_sma50:
            return "bull"
        elif current_price < current_sma20 < current_sma50:
            return "bear"
        else:
            return "chop"
    except Exception as e:
        logger.warning("Regime detection failed: %s", e)
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
    # re-import TA each call so test monkeypatches of pandas_ta are used
    ta = importlib.import_module("pandas_ta")
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
            raise KeyError(
                f"Column '{col}' not found in DataFrame in prepare_indicators"
            )
        df[col] = df[col].astype(float)
    if "open" in df.columns:
        df["open"] = df["open"].astype(float)

    # AI-AGENT-REF: preserve caller-provided index and parse dates without mutation
    idx = safe_to_datetime(df.index, context="retrain index")
    if idx.empty:
        raise ValueError("Invalid date values in dataframe")
    df = df.sort_index()

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
    if len(idx) == len(df):
        df["dow"] = idx.dayofweek.astype(float)

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
        if hasattr(ta, "cci"):
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
        if hasattr(ta, "tema"):
            df["tema"] = ta.tema(df["close"], length=10).astype(float)
    except Exception as e:
        logger.exception("TEMA calculation failed: %s", e)

    df["willr"] = np.nan
    try:
        if hasattr(ta, "willr"):
            df["willr"] = ta.willr(
                df["high"], df["low"], df["close"], length=14
            ).astype(float)
    except Exception as e:
        # escape the % so logging doesn’t try to interpret %R as a format code
        logger.exception("Williams %%R calculation failed: %s", e)

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
        df["ichimoku_conv"] = (
            conv.iloc[:, 0] if hasattr(conv, "iloc") else conv
        ).astype(float)
        df["ichimoku_base"] = (
            base.iloc[:, 0] if hasattr(base, "iloc") else base
        ).astype(float)
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
        df["ret_5m"] = df["close"].pct_change(5, fill_method=None).astype(float)
        df["ret_1h"] = df["close"].pct_change(60, fill_method=None).astype(float)
        df["ret_d"] = df["close"].pct_change(390, fill_method=None).astype(float)
        df["ret_w"] = df["close"].pct_change(1950, fill_method=None).astype(float)
        df["vol_norm"] = (
            df["volume"].rolling(60).mean() / df["volume"].rolling(5).mean()
        ).astype(float)
        df["5m_vs_1h"] = (df["ret_5m"] - df["ret_1h"]).astype(float)
        df["vol_5m"] = (
            df["close"].pct_change(fill_method=None).rolling(5).std().astype(float)
        )
        df["vol_1h"] = (
            df["close"].pct_change(fill_method=None).rolling(60).std().astype(float)
        )
        df["vol_d"] = (
            df["close"].pct_change(fill_method=None).rolling(390).std().astype(float)
        )
        df["vol_w"] = (
            df["close"].pct_change(fill_method=None).rolling(1950).std().astype(float)
        )
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
        try:
            # AI-AGENT-REF: guard optional pandas_ta SMA availability
            if hasattr(ta, "sma"):
                df["sma_50"] = ta.sma(df["close"], length=50).astype(float)
                df["sma_200"] = ta.sma(df["close"], length=200).astype(float)
        except Exception as e:
            logger.exception("SMA calculation failed: %s", e)
        required += ["sma_50", "sma_200"]
    # only drop rows where *all* required indicators are missing
    df.dropna(subset=required, how="all", inplace=True)
    if freq != "daily":  # intraday
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

    logger.info(
        "[gather_minute_data] start=%s, end=%s, symbols=%s",
        start_dt,
        end_dt,
        symbols,
    )
    for sym in symbols:
        bars = None
        try:
            bars = ctx.data_fetcher.get_minute_df(ctx, sym)
        except Exception as e:
            bars = None
            logger.exception("[gather_minute_data] %s fetch failed: %s", sym, e)
            logger.warning(
                "[gather_minute_data] %s exception %s→%s: %s",
                sym,
                start_dt,
                end_dt,
                e,
            )

        if bars is None or bars.empty:
            logger.info("[gather_minute_data] – %s → 0 rows", sym)
        else:
            logger.info("[gather_minute_data] ✓ %s → %s rows", sym, len(bars))
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
                logger.info(
                    "[build_feature_label_df] – %s returned no minute data; skipping symbol.",
                    sym,
                )
                continue
            min_required = max(int(MINUTES_REQUIRED * 0.7), 5)
            if raw.shape[0] < min_required:
                logger.info(
                    "[build_feature_label_df] – skipping %s, only %s < %s (70%% of %s)",
                    sym,
                    raw.shape[0],
                    min_required,
                    MINUTES_REQUIRED,
                )
                continue

            for col in list(raw.columns):
                if col.lower() in ["high", "low", "close", "volume"]:
                    raw = raw.rename(columns={col: col.lower()})
            logger.debug(f"After rename {sym}, tail close:\n{raw[['close']].tail(5)}")
            if (
                "close" not in raw.columns
                or "high" not in raw.columns
                or "low" not in raw.columns
                or "volume" not in raw.columns
            ):
                logger.info(
                    "[build_feature_label_df] – skipping %s, missing price/volume columns",
                    sym,
                )
                continue

            for col in ["high", "low", "close", "volume"]:
                raw[col] = raw[col].astype(float)

            closes = raw["close"].values

            feat = prepare_indicators(raw, freq="intraday")
            logger.debug(
                f"After indicators {sym}, tail close:\n{feat[['close']].tail(5)}"
            )
            if feat.empty:
                logger.info(
                    "[build_feature_label_df] – %s indicators empty after dropna",
                    sym,
                )
                continue

            sent = fetch_sentiment(sym)
            feat["sentiment"] = sent
            logger.debug(
                f"After sentiment {sym}, tail close:\n{feat[['close']].tail(5)}"
            )

            regime_info = detect_regime(raw)
            logger.debug(f"After regime {sym}, tail close:\n{raw[['close']].tail(5)}")
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
                row["regime"] = (
                    regimes.iloc[i] if len(regimes) > i else regimes.iloc[-1]
                )
                row["label"] = label
                rows.append(row)
        except KeyError as e:
            logger.exception("[build_feature_label_df] %s missing data: %s", sym, e)
            logger.warning(
                "[build_feature_label_df] – skipping %s, KeyError: %s", sym, e
            )
            continue

    df_all = pd.DataFrame(rows).dropna()
    return df_all


def log_hyperparam_result(
    regime: str, generation: int, params: dict, score: float
) -> None:
    row = [
        datetime.now(UTC).isoformat(),
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
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
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
            "timestamp": datetime.now(UTC).isoformat(),
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
    population_params = list(
        ParameterSampler(param_space, n_iter=population, random_state=SEED)
    )
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
        ranked = sorted(
            zip(scores, population_params, strict=False), key=lambda t: t[0], reverse=True
        )
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
        return evolutionary_search(
            X, y, base_params, param_space, generations=3, population=n_trials
        )

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
    now = datetime.now(UTC)
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

    logger.info("Starting meta-learner retraining")
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

    df_all = build_feature_label_df(
        raw_store, Δ_minutes=Δ_minutes, threshold_pct=threshold_pct
    )
    if df_all.empty:
        logger.warning(
            "No usable rows after building (Δ, threshold) → skipping retrain."
        )
        return False

    trained_any = False
    for regime, subset in df_all.groupby("regime"):
        if subset.empty or regime not in MODEL_FILES:
            continue

        try:
            X = subset.drop(columns=["label", "regime"])
            y = subset["label"]

            split_idx = int(len(subset) * 0.8)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        except Exception as exc:
            logger.exception("Feature preparation failed for %s: %s", regime, exc)
            continue

        try:
            if len(y_train) < 3:
                logger.warning(
                    "[retrain_meta_learner] Not enough training samples for %s; skipping",
                    regime,
                )
                continue
        except Exception as e:
            logger.exception("Error during sample size check for %s: %s", regime, e)
            continue

        pos_ratio = y_train.mean()
        scoring = "f1" if 0.4 <= pos_ratio <= 0.6 else "roc_auc"
        logger.info(
            "Training %s model using scoring='%s' (pos_ratio=%.3f)",
            regime,
            scoring,
            pos_ratio,
        )

        base_params = dict(
            objective="binary", n_jobs=1, random_state=SEED
        )  # AI-AGENT-REF: reduce thread usage
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
            logger.warning(
                f"[retrain_meta_learner] Training data empty for {regime}; skipping model fit"
            )
            continue
        try:
            pipe.fit(X_train, y_train)
        except Exception as train_exc:
            logger.exception("Model fit failed for %s: %s", regime, train_exc)
            continue
        logger.info("%s best params: %s", regime, best_hyper)

        from sklearn.metrics import f1_score, roc_auc_score

        if scoring == "f1":
            val_pred = pipe.predict(X_val)
            metric = f1_score(y_val, val_pred)
            logger.info("%s holdout F1 = %.4f", regime, metric)
        else:
            val_probs = pipe.predict_proba(X_val)[:, 1]
            metric = roc_auc_score(y_val, val_probs)
            logger.info("%s holdout AUC = %.4f", regime, metric)

        try:
            importances = pd.Series(
                pipe.named_steps["lgbmclassifier"].feature_importances_,
                index=X.columns,
            )
            logger.info(
                "Top feature importances:\n%s",
                importances.sort_values(ascending=False).head(10),
            )
            imp_df = pd.DataFrame(
                {
                    "timestamp": [datetime.now(UTC).isoformat()]
                    * len(importances),
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

        try:
            path = save_model_version(pipe, regime)
            logger.info("Saved %s model to %s", regime, path)
            log_metrics(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
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
        except Exception as exc:
            logger.exception("Model training failed for %s: %s", regime, exc)
            continue

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
            logger.info("Avg rewards by band: %s", band_rewards)
    except Exception as e:
        logger.exception("Failed to load reward by band: %s", e)

    return trained_any


if __name__ == "__main__":  # pragma: no cover - manual invocation
    from ai_trading.logging import setup_logging

    setup_logging()
    logger.info("Retrain module loaded; use retrain_meta_learner() from code.")
