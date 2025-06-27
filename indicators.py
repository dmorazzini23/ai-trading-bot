"""Technical indicator helpers used across the bot."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import jit


def ichimoku_fallback(high: pd.Series, low: pd.Series, close: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple Ichimoku cloud implementation used when pandas_ta is unavailable."""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)

    conv = (high.rolling(9).max() + low.rolling(9).min()) / 2
    base = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((conv + base) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    lagging = close.shift(-26)

    df = pd.DataFrame({
        "ITS_9": conv,
        "IKS_26": base,
        "ISA_26": span_a,
        "ISB_52": span_b,
        "ICS_26": lagging,
    })

    signal = pd.DataFrame(df)
    return df, signal


@jit(nopython=True)
def rsi_numba(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute RSI using a fast numba implementation."""
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed > 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0.0
    rsi = np.zeros_like(prices)
    rsi[:period] = 100.0 - 100.0 / (1.0 + rs)
    up_avg = up
    down_avg = down
    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        up_val = delta if delta > 0 else 0.0
        down_val = -delta if delta < 0 else 0.0
        up_avg = (up_avg * (period - 1) + up_val) / period
        down_avg = (down_avg * (period - 1) + down_val) / period
        rs = up_avg / down_avg if down_avg != 0 else 0.0
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return rsi

