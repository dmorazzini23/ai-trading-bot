"""Technical indicator helpers used across the bot."""

from __future__ import annotations

# AI-AGENT-REF: guard numpy/pandas imports for test environments
try:
    import numpy as np
except ImportError:

    class MockNumpy:
        def array(self, *args, **kwargs):
            return []

        def mean(self, *args, **kwargs):
            return 0.0

        def std(self, *args, **kwargs):
            return 1.0

        def nan(self):
            return float("nan")

        def isnan(self, *args, **kwargs):
            return False

        def zeros(self, *args, **kwargs):
            return []

    np = MockNumpy()

try:
    import pandas as pd
except ImportError:
    from datetime import datetime

    class MockSeries:
        def __init__(self, *args, **kwargs):
            pass

        def mean(self):
            return 0.0

        def std(self):
            return 1.0

    class MockPandas:
        Series = MockSeries
        Timestamp = datetime

    pd = MockPandas()

from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

try:
    from numba import jit
except ImportError:
    # AI-AGENT-REF: numba fallback
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


from typing import Any

_INDICATOR_CACHE: dict[tuple[str, Any], Any] = {}


def ichimoku_fallback(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple Ichimoku cloud implementation used when pandas_ta is unavailable."""
    # AI-AGENT-REF: Add input validation and error handling
    try:
        # Validate inputs
        if len(high) == 0 or len(low) == 0 or len(close) == 0:
            raise ValueError("Input series cannot be empty")

        if len(high) < 52:  # Need at least 52 periods for proper calculation
            raise ValueError("Insufficient data: need at least 52 periods for Ichimoku")

        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)

        # Validate data integrity
        if high.isna().all() or low.isna().all() or close.isna().all():
            raise ValueError("Input series contain only NaN values")

        conv = (high.rolling(9).max() + low.rolling(9).min()) / 2
        base = (high.rolling(26).max() + low.rolling(26).min()) / 2
        span_a = ((conv + base) / 2).shift(26)
        span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        lagging = close.shift(-26)

        df = pd.DataFrame(
            {
                "ITS_9": conv,
                "IKS_26": base,
                "ISA_26": span_a,
                "ISB_52": span_b,
                "ICS_26": lagging,
            }
        )

        signal = pd.DataFrame(df)
        return df, signal
    except (KeyError, ValueError, TypeError, AttributeError):
        # Return empty DataFrames on error to prevent system crash
        empty_df = pd.DataFrame()
        return empty_df, empty_df


@jit(nopython=True)
def _rsi_numba_core(prices_array: np.ndarray, period: int = 14) -> np.ndarray:
    """Core RSI computation using numba."""
    deltas = np.diff(prices_array)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    rsi = np.zeros_like(prices_array)
    if prices_array.size <= period:
        return rsi

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    rs = avg_gain / avg_loss if avg_loss != 0 else 0.0
    rsi[:period] = 100.0 - 100.0 / (1.0 + rs)

    # AI-AGENT-REF: loop retained for sequential smoothing; numba handles speed
    for i in range(period, len(prices_array)):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 0.0
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def rsi_numba(prices, period: int = 14):
    """Compute RSI using a fast numba implementation."""
    # Handle DataFrame/Series input by extracting close prices
    if hasattr(prices, "close"):
        prices_array = prices["close"].values
    elif hasattr(prices, "values"):
        prices_array = prices.values
    else:
        prices_array = np.asarray(prices, dtype=float)

    return _rsi_numba_core(prices_array, period)


# AI-AGENT-REF: new vectorized indicator helpers with caching


@lru_cache(maxsize=128)
def ema(series: tuple[float, ...], period: int) -> pd.Series:
    """Calculate EMA with input validation."""
    try:
        # AI-AGENT-REF: Add input validation
        if period <= 0:
            raise ValueError("Period must be positive")
        if len(series) == 0:
            raise ValueError("Input series cannot be empty")

        s = pd.Series(series)

        # Check for all NaN values
        if s.isna().all():
            raise ValueError("Input series contains only NaN values")

        return s.ewm(span=period, adjust=False).mean()
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        logger.exception(f"Error calculating EMA: {e}")
        # Return empty Series on error
        return pd.Series(dtype=float)


@lru_cache(maxsize=128)
def sma(series: tuple[float, ...], period: int) -> pd.Series:
    """Calculate SMA with input validation."""
    try:
        # AI-AGENT-REF: Add input validation
        if period <= 0:
            raise ValueError("Period must be positive")
        if len(series) == 0:
            raise ValueError("Input series cannot be empty")

        s = pd.Series(series)

        # Check for all NaN values
        if s.isna().all():
            raise ValueError("Input series contains only NaN values")

        return s.rolling(window=period).mean()
    except (KeyError, ValueError, TypeError, AttributeError):
        # Return empty Series on error
        return pd.Series(dtype=float)


def bollinger_bands(x, length: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Calculate Bollinger Bands for given price series."""
    try:
        # AI-AGENT-REF: Add input validation and error handling
        if length <= 0:
            raise ValueError("Length must be positive")
        if num_std < 0:
            raise ValueError("Number of standard deviations must be non-negative")

        if isinstance(x, list | tuple):
            x = pd.Series(x)
        elif hasattr(x, "close"):
            x = x["close"]

        # Validate we have enough data
        if len(x) < length:
            raise ValueError(f"Insufficient data: need at least {length} periods")

        # Check for all NaN values
        if x.isna().all():
            raise ValueError("Input series contains only NaN values")

        sma = x.rolling(window=length).mean()
        std = x.rolling(window=length).std()

        # Handle case where std is 0 (no price movement)
        std = std.fillna(0)

        upper = sma + (std * num_std)
        lower = sma - (std * num_std)

        return pd.DataFrame({"upper": upper, "middle": sma, "lower": lower})
    except (KeyError, ValueError, TypeError, AttributeError):
        # Return empty DataFrame on error to prevent system crash
        return pd.DataFrame(
            {
                "upper": pd.Series(dtype=float),
                "middle": pd.Series(dtype=float),
                "lower": pd.Series(dtype=float),
            }
        )


@lru_cache(maxsize=128)
def rsi(series: tuple[float, ...], period: int = 14) -> pd.Series:
    """Calculate RSI with input validation."""
    try:
        # AI-AGENT-REF: Add input validation
        if period <= 0:
            raise ValueError("Period must be positive")
        if len(series) < period + 1:  # Need at least period+1 for diff calculation
            raise ValueError(f"Insufficient data: need at least {period + 1} values")

        arr = np.asarray(series, dtype=float)

        # Check for all NaN values
        if np.isnan(arr).all():
            raise ValueError("Input series contains only NaN values")

        result = rsi_numba(arr, period)
        return pd.Series(result)
    except (KeyError, ValueError, TypeError, AttributeError):
        # Return empty Series on error
        return pd.Series(dtype=float)


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Calculate Average True Range with input validation."""
    try:
        # AI-AGENT-REF: Add input validation
        if period <= 0:
            raise ValueError("Period must be positive")

        # Validate input series
        for series, name in [(high, "high"), (low, "low"), (close, "close")]:
            if len(series) == 0:
                raise ValueError(f"{name} series cannot be empty")
            if series.isna().all():
                raise ValueError(f"{name} series contains only NaN values")

        # Check all series have same length
        if not (len(high) == len(low) == len(close)):
            raise ValueError("High, low, and close series must have same length")

        if len(high) < period + 1:  # Need period+1 for shift operation
            raise ValueError(f"Insufficient data: need at least {period + 1} periods")

        hl = high - low
        hc = (high - close.shift()).abs()
        lc = (low - close.shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    except (KeyError, ValueError, TypeError, AttributeError):
        # Return empty Series on error
        return pd.Series(dtype=float)


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
def compute_bollinger(
    df: pd.DataFrame, window: int = 20, num_std: int = 2
) -> pd.DataFrame:
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
def calculate_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series]:
    """Return MACD and signal line series."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def calculate_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def calculate_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    typical_price = (high + low + close) / 3
    cum_pv = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_pv / cum_vol


def get_rsi_signal(series: pd.Series | pd.DataFrame, period: int = 14) -> pd.Series:
    """Return normalized RSI signal handling DataFrame or Series input."""
    if isinstance(series, pd.DataFrame):
        # prefer the 'close' column when present, else use the first column
        close_col = series.get("close")
        if close_col is not None:
            series = close_col.astype(float)
        else:
            series = series.iloc[:, 0].astype(float)
    vals = rsi(tuple(series.astype(float)), period)
    return (vals - 50) / 50


def get_atr_trailing_stop(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    period: int = 14,
    multiplier: float = 1.5,
) -> pd.Series:
    atr_series = calculate_atr(high, low, close, period)
    return close - multiplier * atr_series


def cached_atr_trailing_stop(
    symbol: str, df: pd.DataFrame, period: int = 14, multiplier: float = 1.5
) -> pd.Series:
    """Return ATR stop with simple caching by symbol and last timestamp."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    ts = df.index[-1]
    key = (symbol, ts)
    if key in _INDICATOR_CACHE:
        return _INDICATOR_CACHE[key]
    stops = get_atr_trailing_stop(
        df["close"], df["high"], df["low"], period, multiplier
    )
    _INDICATOR_CACHE[key] = stops
    return stops


def get_vwap_bias(
    close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series
) -> pd.Series:
    vwap_series = calculate_vwap(high, low, close, volume)
    bias = close / vwap_series - 1
    return bias


# AI-AGENT-REF: additional indicator utilities for complex strategies
def vwap(prices: np.ndarray, volumes: np.ndarray) -> float:
    """Return the volume weighted average price for prices with validation."""
    try:
        # AI-AGENT-REF: Add input validation
        if len(prices) == 0 or len(volumes) == 0:
            raise ValueError("Prices and volumes arrays cannot be empty")

        if len(prices) != len(volumes):
            raise ValueError("Prices and volumes arrays must have same length")

        if np.isnan(prices).all() or np.isnan(volumes).all():
            raise ValueError("Input arrays contain only NaN values")

        total_volume = np.sum(volumes)
        if total_volume == 0:
            raise ValueError("Total volume cannot be zero")

        return np.sum(prices * volumes) / total_volume
    except (ValueError, TypeError, ZeroDivisionError):
        # Return 0 on error to prevent system crash
        return 0.0


def donchian_channel(
    highs: np.ndarray, lows: np.ndarray, period: int = 20
) -> dict[str, float]:
    """Return Donchian channel bounds using period lookback with validation."""
    try:
        # AI-AGENT-REF: Add input validation
        if period <= 0:
            raise ValueError("Period must be positive")

        if len(highs) == 0 or len(lows) == 0:
            raise ValueError("Highs and lows arrays cannot be empty")

        if len(highs) != len(lows):
            raise ValueError("Highs and lows arrays must have same length")

        if len(highs) < period:
            raise ValueError(f"Insufficient data: need at least {period} periods")

        if np.isnan(highs).all() or np.isnan(lows).all():
            raise ValueError("Input arrays contain only NaN values")

        upper = np.max(highs[-period:])
        lower = np.min(lows[-period:])
        return {"upper": upper, "lower": lower}
    except (ValueError, TypeError, IndexError):
        # Return safe default values on error
        return {"upper": 0.0, "lower": 0.0}


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


def hurst_exponent(ts):
    # AI-AGENT-REF: support DataFrame input and downsample large arrays
    series = ts.iloc[:, 0] if isinstance(ts, pd.DataFrame) else ts
    arr = series.values
    n = len(arr)
    if n > 10000:
        step = max(1, n // 10000)
        arr = arr[::step]
    lags = range(2, 20)
    tau = [np.std(arr[lag:] - arr[:-lag]) for lag in lags]
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
    "cached_atr_trailing_stop",
    "get_vwap_bias",
    "vwap",
    "donchian_channel",
    "obv",
    "stochastic_rsi",
    "hurst_exponent",
]
