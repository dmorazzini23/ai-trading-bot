"""Simple signal generation module for tests."""

import importlib
import logging
import time
from typing import Any, Optional

import pandas as pd
import numpy as np
import requests

logger = logging.getLogger(__name__)


def load_module(name: str) -> Any:
    """Dynamically import a module using :mod:`importlib`."""
    try:
        return importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - dynamic import may fail
        logger.warning("Failed to import %s: %s", name, exc)
        return None


def _fetch_api(url: str, retries: int = 3, delay: float = 1.0) -> dict:
    """Fetch JSON from an API with simple retry logic."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # pragma: no cover - network may be mocked
            logger.warning(
                "API request failed (%s/%s): %s", attempt, retries, exc
            )
            time.sleep(delay)
    return {}


def generate(ctx: Any | None = None) -> int:
    """Placeholder generate function used in tests."""

    return 0


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
        if close_prices is None or len(close_prices) < min_len:
            logger.warning(
                "Insufficient data for MACD calculation: length=%s",
                len(close_prices) if close_prices is not None else "None",
            )
            return None
        if close_prices.isna().any() or np.isinf(close_prices).any():
            logger.warning("MACD input data contains NaN or Inf")
            return None

        fast_ema = close_prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = close_prices.ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        macd_df = pd.DataFrame(
            {"macd": macd_line, "signal": signal_line, "histogram": histogram}
        )

        if macd_df.isnull().values.any():
            logger.warning("MACD calculation returned NaNs in the result")
            return None

        return macd_df

    except Exception as exc:  # pragma: no cover - defensive
        logger.error("MACD calculation failed with exception: %s", exc, exc_info=True)
        return None


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

    if "close" not in data.columns:
        raise ValueError("Input data missing 'close' column")

    macd_df = calculate_macd(data["close"])
    if macd_df is None:
        logger.warning("MACD indicator calculation failed, returning None")
        raise ValueError("MACD calculation failed")

    for col in macd_df.columns:
        data[col] = macd_df[col]

    # Additional indicators can be added here using similar defensive checks

    return data


def generate_signal(df: pd.DataFrame, column: str) -> pd.Series:
    if df is None or df.empty:
        logger.error("Dataframe is None or empty in generate_signal")
        return pd.Series(dtype=float)

    if column not in df.columns:
        logger.error(f"Required column '{column}' not found in dataframe")
        return pd.Series(dtype=float)

    try:
        signal = pd.Series(0, index=df.index)
        signal[df[column] > 0] = 1
        signal[df[column] < 0] = -1
        return signal.fillna(0)
    except Exception as e:
        logger.error(f"Exception generating signal: {e}", exc_info=True)
        return pd.Series(dtype=float)
