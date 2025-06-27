"""Simple signal generation module for tests."""

import importlib
import logging
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple rolling mean using cumulative sum for speed."""
    if window <= 0:
        raise ValueError("window must be positive")
    arr = np.asarray(arr, dtype=float)
    if arr.size < window:
        return np.array([], dtype=float)
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)

try:
    from hmmlearn.hmm import GaussianHMM
except Exception:  # pragma: no cover - optional dependency
    GaussianHMM = None

logger = logging.getLogger(__name__)


def load_module(name: str) -> Any:
    """Dynamically import a module using :mod:`importlib`."""
    try:
        return importlib.import_module(name)
    except ImportError as exc:  # pragma: no cover - dynamic import may fail
        logger.warning("Failed to import %s: %s", name, exc)
        return None


def _fetch_api(url: str, retries: int = 3, delay: float = 1.0) -> dict:
    """Fetch JSON from an API with simple retry logic."""
    # TODO: check loop for numpy replacement
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            return resp.json()
        except (
            requests.RequestException
        ) as exc:  # pragma: no cover - network may be mocked
            logger.warning("API request failed (%s/%s): %s", attempt, retries, exc)
            time.sleep(delay)
    return {}


def generate() -> int:
    """Placeholder generate function used in tests."""

    return 0


def _validate_macd_input(close_prices: pd.Series, min_len: int) -> bool:
    if close_prices is None or len(close_prices) < min_len:
        logger.warning(
            "Insufficient data for MACD calculation: length=%s",
            len(close_prices) if close_prices is not None else "None",
        )
        return False
    if close_prices.isna().any() or np.isinf(close_prices).any():
        logger.warning("MACD input data contains NaN or Inf")
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

        macd_df = _compute_macd_df(
            close_prices, fast_period, slow_period, signal_period
        )

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
    # TODO: check loop for numpy replacement
    for col in ("macd", "signal", "histogram"):
        series = macd_df.get(col)
        if series is None:
            raise ValueError(f"MACD output missing column '{col}'")
        data[col] = series.astype(float)
    return data


def prepare_indicators(data: pd.DataFrame) -> pd.DataFrame:
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
    data = _apply_macd(data)
    # Additional indicators can be added here using similar defensive checks
    return data


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
