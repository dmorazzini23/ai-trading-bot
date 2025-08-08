"""Feature engineering for reinforcement learning.

This module provides helpers to compute flattened feature vectors from
historical price data for reinforcement learning.  The goal is to
capture multiple dimensions of market behaviour – momentum, volatility
and price-volume bias – so that the RL policy can make more informed
decisions.  Each feature vector has a fixed length ``window * num_features``
where ``window`` is the number of lookback periods and ``num_features``
is the number of indicators per period.  The default configuration uses
six features per period:

1. **Returns** – Normalized percentage changes of closing prices, capturing momentum.
2. **RSI** – Normalized relative strength index values (centered around zero), providing an oscillator measure of overbought/oversold conditions.
3. **ATR** – Average true range, measuring historical volatility.
4. **VWAP bias** – Ratio of price to volume-weighted average price minus one, indicating whether the closing price is above or below the VWAP.
5. **Bollinger position** – Relative position of the last close within its Bollinger Bands (normalized between −1 and 1).
6. **OBV (On-Balance Volume)** – Normalized On-Balance Volume, measuring buying/selling pressure via volume.

These features are concatenated in order and padded with zeros when insufficient history is available.  Additional indicators (e.g., macro data) can be appended by extending the return array.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from ai_trading.indicators import (  # type: ignore
        calculate_atr,
        get_rsi_signal,
        get_vwap_bias,
        bollinger_bands,
        obv,
    )


def compute_features(df: pd.DataFrame | None, window: int = 10) -> np.ndarray:
    """Return a flattened feature vector for a given symbol.

    Parameters
    ----------
    df : pandas.DataFrame or None
        A DataFrame containing at least a ``close`` column. ``high`` and
        ``low`` columns are optional but used when present. If ``df`` is
        ``None`` or empty, a zero-vector of length ``window * num_features`` is
        returned.
    window : int, optional
        The lookback window to compute features for (default: ``10``).

    Returns
    -------
    numpy.ndarray
        A 1-D numpy array of length ``window * num_features`` containing recent
        returns and technical indicators.
    """
    # Number of distinct indicators per period: returns, RSI, ATR, VWAP bias, Bollinger position, OBV
    num_features = 6
    total_len = window * num_features
    if df is None or df.empty or "close" not in df.columns:
        return np.zeros(total_len, dtype=np.float32)
    try:
        close = df["close"].astype(float)
        pct_change = close.pct_change().fillna(0.0)
        returns = pct_change.tail(window)
        rsi_series = get_rsi_signal(close, period=14).tail(window)
        # ATR requires high and low; fallback to zeros
        if {"high", "low"}.issubset(df.columns):
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            atr_series = (
                calculate_atr(high, low, close, period=14).fillna(0.0).tail(window)
            )
        else:
            atr_series = pd.Series(0.0, index=close.index).tail(window)

        # VWAP bias requires high, low and volume; fallback to zeros if missing
        if {"high", "low", "volume"}.issubset(df.columns):
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            volume = df["volume"].astype(float)
            vwap_bias = get_vwap_bias(close, high, low, volume).fillna(0.0).tail(window)
        else:
            vwap_bias = pd.Series(0.0, index=close.index).tail(window)

        # Compute Bollinger band position: normalized between −1 and 1 (close relative to upper/lower bands)
        if {"high", "low"}.issubset(df.columns):
            bb = bollinger_bands(close, length=20, num_std=2.0)
            bb_upper = bb["upper"].tail(window).fillna(method="bfill")
            bb_lower = bb["lower"].tail(window).fillna(method="bfill")
            bb_mid = bb["middle"].tail(window).fillna(method="bfill")
            # Avoid division by zero by adding epsilon to denominator
            bb_range = (bb_upper - bb_lower).replace(0.0, np.nan).fillna(method="bfill")
            bollinger_pos = (
                ((close.tail(window) - bb_mid) / bb_range).clip(-1.0, 1.0).fillna(0.0)
            )
        else:
            bollinger_pos = pd.Series(0.0, index=close.index).tail(window)

        # Compute OBV (On-Balance Volume) normalized to [-1, 1]
        if {"volume"}.issubset(df.columns):
            obv_series = pd.Series(
                obv(
                    close.tail(window + 1).to_numpy(),
                    df["volume"].astype(float).tail(window + 1).to_numpy(),
                ),
                index=close.tail(window + 1).index,
            )
            # Normalize OBV over the window
            obv_norm = (obv_series - obv_series.mean()) / (obv_series.std() + 1e-8)
            obv_norm = obv_norm.tail(window).fillna(0.0).clip(-1.0, 1.0)
        else:
            obv_norm = pd.Series(0.0, index=close.index).tail(window)

        # Concatenate all series into one feature vector
        feat = np.concatenate(
            [
                returns.to_numpy(dtype=np.float32),
                rsi_series.to_numpy(dtype=np.float32),
                atr_series.to_numpy(dtype=np.float32),
                vwap_bias.to_numpy(dtype=np.float32),
                bollinger_pos.to_numpy(dtype=np.float32),
                obv_norm.to_numpy(dtype=np.float32),
            ]
        )
        if feat.size < total_len:
            feat = np.pad(feat, (0, total_len - feat.size), constant_values=0.0)
        return feat[:total_len]
    except Exception:
        return np.zeros(total_len, dtype=np.float32)
