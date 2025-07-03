"""Technical indicator helpers used across the bot."""
from __future__ import annotations

import numpy as np
import pandas as pd
from functools import lru_cache
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
    # TODO: check loop for numpy replacement
    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        up_val = delta if delta > 0 else 0.0
        down_val = -delta if delta < 0 else 0.0
        up_avg = (up_avg * (period - 1) + up_val) / period
        down_avg = (down_avg * (period - 1) + down_val) / period
        rs = up_avg / down_avg if down_avg != 0 else 0.0
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


# AI-AGENT-REF: new vectorized indicator helpers with caching

@lru_cache(maxsize=128)
def ema(series: tuple[float, ...], period: int) -> pd.Series:
    s = pd.Series(series)
    return s.ewm(span=period, adjust=False).mean()


@lru_cache(maxsize=128)
def sma(series: tuple[float, ...], period: int) -> pd.Series:
    s = pd.Series(series)
    return s.rolling(window=period).mean()


def bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    ma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std(ddof=0)
    upper = ma + num_std * std
    lower = ma - num_std * std
    return pd.DataFrame({"bb_upper": upper, "bb_lower": lower, "bb_mid": ma})


@lru_cache(maxsize=128)
def rsi(series: tuple[float, ...], period: int = 14) -> pd.Series:
    arr = np.asarray(series, dtype=float)
    return pd.Series(rsi_numba(arr, period))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def mean_reversion_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std(ddof=0)
    return (series - rolling_mean) / rolling_std

# AI-AGENT-REF: helper to compute multiple EMAs across common horizons
def compute_ema(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
    periods = periods or [5, 20, 50, 200]
    for p in periods:
        df[f"EMA_{p}"] = df["close"].ewm(span=p, adjust=False).mean()
    return df


# AI-AGENT-REF: helper to compute multiple SMAs across common horizons
def compute_sma(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
    periods = periods or [5, 20, 50, 200]
    for p in periods:
        df[f"SMA_{p}"] = df["close"].rolling(window=p).mean()
    return df


# AI-AGENT-REF: compute standard Bollinger Bands and width
def compute_bollinger(df: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    df["MB"] = df["close"].rolling(window=window).mean()
    df["STD"] = df["close"].rolling(window=window).std()
    df["UB"] = df["MB"] + (num_std * df["STD"])
    df["LB"] = df["MB"] - (num_std * df["STD"])
    df["BollingerWidth"] = df["UB"] - df["LB"]
    return df


# AI-AGENT-REF: compute ATR values for several lookback periods
def compute_atr(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
    periods = periods or [14, 50]
    for p in periods:
        tr = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                (df["high"] - df["close"].shift()).abs(),
                (df["low"] - df["close"].shift()).abs(),
            ),
        )
        df[f"TR_{p}"] = tr
        df[f"ATR_{p}"] = tr.rolling(window=p).mean()
    return df


# AI-AGENT-REF: additional indicator utilities
def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    """Return MACD and signal line series."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical_price = (high + low + close) / 3
    cum_pv = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_pv / cum_vol


def get_rsi_signal(series: pd.Series, period: int = 14) -> pd.Series:
    vals = rsi(tuple(series.astype(float)), period)
    return (vals - 50) / 50


def get_atr_trailing_stop(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 14, multiplier: float = 1.5) -> pd.Series:
    atr_series = calculate_atr(high, low, close, period)
    return close - multiplier * atr_series


def get_vwap_bias(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series) -> pd.Series:
    vwap_series = calculate_vwap(high, low, close, volume)
    bias = close / vwap_series - 1
    return bias

# AI-AGENT-REF: additional indicator utilities for complex strategies
def vwap(prices: np.ndarray, volumes: np.ndarray) -> float:
    """Return the volume weighted average price for ``prices``."""
    return np.sum(prices * volumes) / np.sum(volumes)


def donchian_channel(highs: np.ndarray, lows: np.ndarray, period: int = 20) -> dict[str, float]:
    """Return Donchian channel bounds using ``period`` lookback."""
    upper = np.max(highs[-period:])
    lower = np.min(lows[-period:])
    return {"upper": upper, "lower": lower}


def obv(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """Return On-Balance Volume (OBV) series."""
    obv_vals = [0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv_vals.append(obv_vals[-1] + volumes[i])
        elif closes[i] < closes[i - 1]:
            obv_vals.append(obv_vals[-1] - volumes[i])
        else:
            obv_vals.append(obv_vals[-1])
    return np.array(obv_vals)


def stochastic_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Return simplified stochastic RSI as an array aligned with ``prices``."""
    deltas = np.diff(prices)
    ups = np.where(deltas > 0, deltas, 0)
    downs = -np.where(deltas < 0, deltas, 0)
    rs = np.sum(ups[-period:]) / (np.sum(downs[-period:]) + 1e-9)
    rsi_val = 100 - (100 / (1 + rs))
    return np.array([rsi_val] * len(prices))


def hurst_exponent(ts: np.ndarray) -> float:
    """Estimate the Hurst exponent for ``ts``."""
    lags = range(2, 100)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return 2.0 * poly[0]


__all__ = [
    "ichimoku_fallback",
    "rsi_numba",
    "ema",
    "sma",
    "bollinger_bands",
    "compute_ema",
    "compute_sma",
    "compute_bollinger",
    "compute_atr",
    "rsi",
    "atr",
    "mean_reversion_zscore",
    "calculate_macd",
    "calculate_atr",
    "calculate_vwap",
    "get_rsi_signal",
    "get_atr_trailing_stop",
    "get_vwap_bias",
    "vwap",
    "donchian_channel",
    "obv",
    "stochastic_rsi",
    "hurst_exponent",
]

