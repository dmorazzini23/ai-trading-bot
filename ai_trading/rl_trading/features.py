"""Feature engineering for reinforcement learning.

This module provides a simple function to compute a flattened feature
vector from historical price data.  The RL agent expects a 1-D
observation per symbol consisting of recent returns and technical
indicators.  The default window size is 10 periods, and three types
of features are included by default:

1. **Returns** – Normalized percentage change of closing prices.
2. **RSI** – Normalized relative strength index values (centered around
   zero) derived from the closing price series.
3. **ATR** – The average true range, measuring volatility.  When
   ``high`` and ``low`` columns are unavailable, zeros are used.

This function can be extended to include additional indicators (e.g.,
Bollinger Bands, VWAP bias, OBV) by concatenating the corresponding
values into the return array.  All inputs are padded to ensure a
consistent length of ``window * num_features``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import calculate_atr, get_rsi_signal


def compute_features(df: pd.DataFrame | None, window: int = 10) -> np.ndarray:
    """Return a flattened feature vector for a given symbol.

    Parameters
    ----------
    df : pandas.DataFrame or None
        A DataFrame containing at least a ``close`` column. ``high`` and
        ``low`` columns are optional but used when present. If ``df`` is
        ``None`` or empty, a zero-vector of length ``window * 3`` is
        returned.
    window : int, optional
        The lookback window to compute features for (default: ``10``).

    Returns
    -------
    numpy.ndarray
        A 1-D numpy array of length ``window * 3`` containing recent
        returns, RSI values and ATR values.
    """
    num_features = 3  # returns, RSI, ATR
    total_len = window * num_features
    if df is None or df.empty or "close" not in df.columns:
        return np.zeros(total_len, dtype=np.float32)
    try:
        close = df["close"].astype(float)
        pct_change = close.pct_change().fillna(0.0)
        returns = pct_change.tail(window)
        rsi_series = get_rsi_signal(close, period=14).tail(window)
        if {"high", "low"}.issubset(df.columns):
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            atr_series = calculate_atr(high, low, close, period=14).fillna(0.0).tail(window)
        else:
            atr_series = pd.Series(0.0, index=close.index).tail(window)
        feat = np.concatenate([
            returns.to_numpy(dtype=np.float32),
            rsi_series.to_numpy(dtype=np.float32),
            atr_series.to_numpy(dtype=np.float32),
        ])
        if feat.size < total_len:
            feat = np.pad(feat, (0, total_len - feat.size), constant_values=0.0)
        return feat[:total_len]
    except Exception:
        return np.zeros(total_len, dtype=np.float32)
