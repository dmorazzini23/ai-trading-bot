"""Simple signal generation module for tests."""

import importlib
import logging
import os
import time
from typing import Any, Optional
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime, timezone

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:  # pragma: no cover - optional dependency
    GaussianHMM = None

from indicators import rsi, atr, mean_reversion_zscore

# Cache the last computed signal matrix to avoid recomputation
_LAST_SIGNAL_BAR: pd.Timestamp | None = None
_LAST_SIGNAL_MATRIX: pd.DataFrame | None = None

def get_utcnow():
    return datetime.now(timezone.utc)

# AI-AGENT-REF: safe close retrieval for pipelines
def robust_signal_price(df: pd.DataFrame) -> float:
    try:
        return df['close'].iloc[-1]
    except Exception:
        return 1e-3


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple rolling mean using cumulative sum for speed."""
    if window <= 0:
        raise ValueError("window must be positive")
    arr = np.asarray(arr, dtype=float)
    if arr.size < window:
        return np.array([], dtype=float)
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)


logger = logging.getLogger(__name__)


def load_module(name):
    if not isinstance(name, str):
        logger.warning("Skipping load_module on non-string: %s", type(name))
        return None
    return importlib.import_module(name)


def _fetch_api(url: str, retries: int = 3, delay: float = 1.0) -> dict:
    """Fetch JSON from an API with simple retry logic and backoff."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:  # pragma: no cover - network may be mocked
            logger.warning(
                "API request failed (%s/%s): %s", attempt, retries, exc
            )
            time.sleep(delay * attempt)
    return {}


def generate() -> int:
    """Placeholder generate function used in tests."""

    return 0


def _validate_macd_input(close_prices, min_len):
    if close_prices.isna().any() or np.isinf(close_prices).any():
        return False
    if len(close_prices) < min_len:
        return False
    return True


def _compute_macd_df(
    close_prices: pd.Series,
    fast_period: int,
    slow_period: int,
    signal_period: int,
) -> pd.DataFrame:
    fast_ema = close_prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close_prices.ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram}
    )


@lru_cache(maxsize=128)
def _cached_macd(
    prices_tuple: tuple,
    fast_period: int,
    slow_period: int,
    signal_period: int,
) -> pd.DataFrame:
    series = pd.Series(prices_tuple)
    return _compute_macd_df(series, fast_period, slow_period, signal_period)


def calculate_macd(
    close_prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Optional[pd.DataFrame]:
    """Calculate MACD indicator values with validation.

    Parameters
    ----------
    close_prices : pd.Series
        Series of closing prices.
    fast_period : int
        Fast EMA period. Defaults to ``12``.
    slow_period : int
        Slow EMA period. Defaults to ``26``.
    signal_period : int
        Signal line EMA period. Defaults to ``9``.

    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame containing ``macd``, ``signal`` and ``histogram`` columns or
        ``None`` if the calculation fails.
    """

    try:
        min_len = slow_period + signal_period
        if not _validate_macd_input(close_prices, min_len):
            return None

        tup = tuple(map(float, close_prices.dropna().tolist()))
        macd_df = _cached_macd(tup, fast_period, slow_period, signal_period)

        if macd_df.isnull().values.any():
            logger.warning("MACD calculation returned NaNs in the result")
            return None

        return macd_df

    except (ValueError, TypeError) as exc:  # pragma: no cover - defensive
        logger.error("MACD calculation failed with exception: %s", exc, exc_info=True)
        return None


def _validate_input_df(data: pd.DataFrame) -> None:
    if data is None or not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a DataFrame")
    if "close" not in data.columns:
        raise ValueError("Input data missing 'close' column")


def _apply_macd(data: pd.DataFrame) -> pd.DataFrame:
    macd_df = calculate_macd(data["close"])
    if macd_df is None or macd_df.empty:
        logger.warning("MACD indicator calculation failed, returning None")
        raise ValueError("MACD calculation failed")
    missing = [c for c in ("macd", "signal", "histogram") if c not in macd_df]
    if missing:
        raise ValueError(f"MACD output missing column(s) {missing}")
    data[["macd", "signal", "histogram"]] = macd_df[["macd", "signal", "histogram"]].astype(float)
    logger.debug(
        f"After MACD {data.columns[0] if not data.empty else ''}, tail close:\n{data[['close']].tail(5)}"
    )
    return data


def prepare_indicators(data: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    """Prepare indicator columns for a trading strategy.

    Parameters
    ----------
    data : pd.DataFrame
        Market data containing at least a ``close`` column.

    Returns
    -------
    pd.DataFrame
        Data enriched with indicator columns.

    Raises
    ------
    ValueError
        If the MACD indicator fails to calculate or ``close`` column is missing.
    """

    _validate_input_df(data)
    cache_path = Path(f"cache_{ticker}.parquet") if ticker else None
    if os.getenv("DISABLE_PARQUET"):
        cache_path = None
    if cache_path and cache_path.exists():
        return pd.read_parquet(cache_path)

    data = _apply_macd(data.copy())
    logger.debug(
        f"After prepare_macd {ticker or ''}, tail close:\n{data[['close']].tail(5)}"
    )
    if cache_path:
        try:
            data.to_parquet(cache_path, engine="pyarrow")
        except OSError:
            pass

    # Additional indicators can be added here using similar defensive checks
    return data


def prepare_indicators_parallel(symbols, data, max_workers=None):
    """Run :func:`prepare_indicators` over ``symbols`` concurrently."""
    if os.getenv("DISABLE_PARQUET"):
        return
    workers = max_workers or min(4, len(symbols))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(lambda t: prepare_indicators(data[t], t), symbols))

def generate_signal(df: pd.DataFrame, column: str) -> pd.Series:
    if df is None or df.empty:
        logger.error("Dataframe is None or empty in generate_signal")
        return pd.Series(dtype=float)

    if column not in df.columns:
        logger.error("Required column '%s' not found in dataframe", column)
        return pd.Series(dtype=float)

    try:
        values = df[column].to_numpy()
        signal = np.sign(values)
        return pd.Series(signal, index=df.index).fillna(0).astype(int)
    except (ValueError, TypeError) as exc:
        logger.error("Exception generating signal: %s", exc, exc_info=True)
        return pd.Series(dtype=float)


def detect_market_regime_hmm(df: pd.DataFrame, n_states: int = 3) -> pd.DataFrame:
    """Annotate ``df`` with hidden Markov market regimes."""
    if GaussianHMM is None:
        df["Regime"] = np.nan
        return df
    returns = np.log(df["Close"]).diff().dropna().values.reshape(-1, 1)
    if len(returns) < n_states * 10:
        df["Regime"] = np.nan
        return df
    try:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=1000,
            random_state=42,
        )
        model.fit(returns)
        hidden_states = model.predict(returns)
        df["Regime"] = np.append([np.nan], hidden_states)
    except Exception as e:  # pragma: no cover - hmmlearn may fail
        logger.warning("HMM failed: %s", e)
        df["Regime"] = np.nan
    return df


def compute_signal_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return a matrix of z-scored indicator signals."""

    if df is None or df.empty:
        return pd.DataFrame()
    global _LAST_SIGNAL_BAR, _LAST_SIGNAL_MATRIX
    last_bar = df.index[-1] if not df.empty else None
    if last_bar is not None and last_bar == _LAST_SIGNAL_BAR:
        # AI-AGENT-REF: reuse previously computed indicators for same bar
        return _LAST_SIGNAL_MATRIX.copy() if _LAST_SIGNAL_MATRIX is not None else pd.DataFrame()
    required = {"close", "high", "low"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    macd_df = calculate_macd(df["close"])
    rsi_series = rsi(tuple(df["close"].fillna(method="ffill").astype(float)), 14)
    sma_diff = df["close"] - df["close"].rolling(20).mean()
    atr_series = atr(df["high"], df["low"], df["close"], 14)
    atr_move = df["close"].diff() / atr_series.replace(0, np.nan)
    mean_rev = mean_reversion_zscore(df["close"], 20)

    def _z(series: pd.Series) -> pd.Series:
        return (series - series.rolling(20).mean()) / series.rolling(20).std(ddof=0)

    matrix = pd.DataFrame(index=df.index)
    if macd_df is not None and not macd_df.empty:
        matrix["macd"] = _z(macd_df["macd"])
    matrix["rsi"] = _z(rsi_series)
    matrix["sma_diff"] = _z(sma_diff)
    matrix["atr_move"] = _z(atr_move)
    matrix["mean_rev_z"] = mean_rev
    matrix = matrix.dropna(how="all")
    _LAST_SIGNAL_BAR = last_bar
    _LAST_SIGNAL_MATRIX = matrix.copy()
    return matrix


def ensemble_vote_signals(signal_matrix: pd.DataFrame) -> pd.Series:
    """Return voting-based entry signals from ``signal_matrix``."""

    if signal_matrix is None or signal_matrix.empty:
        return pd.Series(dtype=int)
    pos = (signal_matrix > 0.5).sum(axis=1)
    neg = (signal_matrix < -0.5).sum(axis=1)
    votes = np.where(pos >= 2, 1, np.where(neg >= 2, -1, 0))
    return pd.Series(votes, index=signal_matrix.index)


def classify_regime(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Classify each row as 'trend' or 'mean_revert' based on volatility."""

    if df is None or df.empty or "close" not in df:
        return pd.Series(dtype=object)
    returns = df["close"].pct_change()
    vol = returns.rolling(window).std()
    med = vol.rolling(window).median()
    dev = vol.rolling(window).std()
    regime = np.where(vol > med + dev, "trend", "mean_revert")
    return pd.Series(regime, index=df.index)


# AI-AGENT-REF: ensemble decision using multiple indicator columns
def generate_ensemble_signal(df: pd.DataFrame) -> int:
    signals = []
    if df.get("EMA_5", pd.Series()).iloc[-1] > df.get("EMA_20", pd.Series()).iloc[-1]:
        signals.append(1)
    if df.get("SMA_50", pd.Series()).iloc[-1] > df.get("SMA_200", pd.Series()).iloc[-1]:
        signals.append(1)
    if df.get("close", pd.Series()).iloc[-1] > df.get("UB", pd.Series()).iloc[-1]:
        signals.append(-1)
    if df.get("close", pd.Series()).iloc[-1] < df.get("LB", pd.Series()).iloc[-1]:
        signals.append(1)
    avg_signal = np.mean(signals) if signals else 0
    if avg_signal > 0.5:
        return 1
    if avg_signal < -0.5:
        return -1
    return 0
