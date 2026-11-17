from __future__ import annotations
from typing import TYPE_CHECKING

from ai_trading.utils.lazy_imports import load_pandas

if TYPE_CHECKING:
    import pandas as pd


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    pd = load_pandas()
    import numpy as np

    close = pd.to_numeric(close, errors='coerce')
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    au = up.ewm(alpha=1 / length, adjust=False).mean()
    ad = dn.ewm(alpha=1 / length, adjust=False).mean()
    rs = au / (ad + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    pd = load_pandas()

    high = pd.to_numeric(high, errors='coerce')
    low = pd.to_numeric(low, errors='coerce')
    close = pd.to_numeric(close, errors='coerce')
    pc = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()

def vwap_bias(close, high, low, volume, length: int = 20) -> pd.Series:
    pd = load_pandas()
    import numpy as np

    close = pd.to_numeric(close, errors='coerce')
    high = pd.to_numeric(high, errors='coerce')
    low = pd.to_numeric(low, errors='coerce')
    volume = pd.to_numeric(volume, errors='coerce')
    typical = (high + low + close) / 3.0
    pv = (typical * volume).rolling(length, min_periods=1).sum()
    vv = volume.rolling(length, min_periods=1).sum() + 1e-12
    vwap = pv / vv
    diff = (close - vwap) / (np.abs(vwap) + 1e-12)
    return np.tanh(diff)

def bollinger_position(close: pd.Series, length: int = 20, nstd: float = 2.0) -> pd.Series:
    pd = load_pandas()
    import numpy as np

    close = pd.to_numeric(close, errors='coerce')
    ma = close.rolling(length, min_periods=1).mean()
    sd = close.rolling(length, min_periods=1).std().fillna(0.0)
    upper = ma + nstd * sd
    lower = ma - nstd * sd
    rng = (upper - lower).replace(0.0, np.nan)
    pos = (close - ma) / (rng + 1e-12)
    return pos.clip(-1.0, 1.0).fillna(0.0)

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    pd = load_pandas()
    import numpy as np

    close = pd.to_numeric(close, errors='coerce')
    volume = pd.to_numeric(volume, errors='coerce').fillna(0.0)
    sign = np.sign(close.diff().fillna(0.0))
    return (sign * volume).cumsum().fillna(0.0)
