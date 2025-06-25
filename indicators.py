"""Technical indicator helpers used across the bot."""
from __future__ import annotations

import pandas as pd


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

