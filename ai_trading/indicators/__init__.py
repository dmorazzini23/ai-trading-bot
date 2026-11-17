"""Technical indicator helpers used across the bot."""
from __future__ import annotations

from ai_trading.logging import get_logger
from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any, TypeAlias, cast
import numpy as np
from ai_trading.utils.lazy_imports import load_pandas
from .manager import IndicatorManager, Indicator

logger = get_logger(__name__)

# Lazy pandas proxy
pd = load_pandas()
_PANDAS_DATAFRAME_TYPE = pd.DataFrame

if TYPE_CHECKING:
    import pandas as _pd

    PDDataFrame: TypeAlias = _pd.DataFrame
    PDSeries: TypeAlias = _pd.Series
else:
    PDDataFrame = Any
    PDSeries = Any

try:  # optional numba import
    import numba as _numba  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    _numba = None
    logger.info('NUMBA_MISSING', extra={'hint': 'pip install numba'})
_numba_jit = getattr(_numba, "jit", None) if _numba is not None else None

def jit(*args, **kwargs):
    """Lazily apply numba JIT based on configuration."""

    def decorator(func):
        from functools import wraps

        @wraps(func)
        def wrapper(*f_args, **f_kwargs):
            from ai_trading.config.settings import get_settings
            if _numba_jit and getattr(get_settings(), 'enable_numba_optimization', False):
                return _numba_jit(*args, **kwargs)(func)(*f_args, **f_kwargs)
            return func(*f_args, **f_kwargs)
        return wrapper
    return decorator

_INDICATOR_CACHE: dict[tuple[str, Any], Any] = {}


@jit(nopython=True)
def _rsi_numba_core(prices_array: np.ndarray, period: int=14) -> np.ndarray:
    """Core RSI computation using numba."""
    deltas = np.diff(prices_array)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    rsi = np.zeros_like(prices_array)
    if prices_array.size <= period:
        return cast(np.ndarray, rsi)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    rs = avg_gain / avg_loss if avg_loss != 0 else 0.0
    rsi[:period] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(period, len(prices_array)):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 0.0
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return cast(np.ndarray, rsi)

def rsi_numba(prices, period: int=14):
    """Compute RSI using a fast numba implementation."""
    if hasattr(prices, 'close'):
        prices_array = prices['close'].values
    elif hasattr(prices, 'values'):
        prices_array = prices.values
    else:
        prices_array = np.asarray(prices, dtype=float)
    return _rsi_numba_core(prices_array, period)

@lru_cache(maxsize=128)
def ema(series: Sequence[float], period: int) -> PDSeries:
    """Calculate EMA with input validation."""
    try:
        if period <= 0:
            raise ValueError('Period must be positive')
        if len(series) == 0:
            raise ValueError('Input series cannot be empty')
        s = pd.Series(series)
        if s.isna().all():
            raise ValueError('Input series contains only NaN values')
        return s.ewm(span=period, adjust=False).mean()
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        logger.exception(f'Error calculating EMA: {e}')
        return pd.Series(dtype=float)

@lru_cache(maxsize=128)
def sma(series: Sequence[float], period: int) -> PDSeries:
    """Calculate SMA with input validation."""
    try:
        if period <= 0:
            raise ValueError('Period must be positive')
        if len(series) == 0:
            raise ValueError('Input series cannot be empty')
        s = pd.Series(series)
        if s.isna().all():
            raise ValueError('Input series contains only NaN values')
        return s.rolling(window=period).mean()
    except (KeyError, ValueError, TypeError, AttributeError):
        return pd.Series(dtype=float)


@lru_cache(maxsize=128)
def rsi(series: Sequence[float], period: int=14) -> PDSeries:
    """Calculate RSI with input validation."""
    try:
        if period <= 0:
            raise ValueError('Period must be positive')
        if len(series) < period + 1:
            raise ValueError(f'Insufficient data: need at least {period + 1} values')
        arr = np.asarray(series, dtype=float)
        if np.isnan(arr).all():
            raise ValueError('Input series contains only NaN values')
        result = rsi_numba(arr, period)
        return pd.Series(result)
    except (KeyError, ValueError, TypeError, AttributeError):
        return pd.Series(dtype=float)

def atr(high: PDSeries, low: PDSeries, close: PDSeries, period: int=14) -> PDSeries:
    """Calculate Average True Range with input validation."""
    try:
        if period <= 0:
            raise ValueError('Period must be positive')
        for series, name in [(high, 'high'), (low, 'low'), (close, 'close')]:
            if len(series) == 0:
                raise ValueError(f'{name} series cannot be empty')
            if series.isna().all():
                raise ValueError(f'{name} series contains only NaN values')
        if not len(high) == len(low) == len(close):
            raise ValueError('High, low, and close series must have same length')
        if len(high) < period + 1:
            raise ValueError(f'Insufficient data: need at least {period + 1} periods')
        hl = high - low
        hc = (high - close.shift()).abs()
        lc = (low - close.shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    except (KeyError, ValueError, TypeError, AttributeError):
        return pd.Series(dtype=float)

def mean_reversion_zscore(series: PDSeries, window: int=20) -> PDSeries:
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std(ddof=0)
    return (series - rolling_mean) / rolling_std


def compute_sma(df: PDDataFrame, periods: list[int] | None=None) -> PDDataFrame:
    periods = periods or [5, 20, 50, 200]
    for p in periods:
        df[f'SMA_{p}'] = df['close'].rolling(window=p).mean()
    return df


def compute_atr(df: PDDataFrame, periods: list[int] | None=None) -> PDDataFrame:
    """
    Compute the Average True Range (ATR) for one or more lookback periods.
    """
    periods = periods or [14, 50]
    for p in periods:
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                (df['high'] - df['close'].shift()).abs(),
                (df['low'] - df['close'].shift()).abs(),
            ),
        )
        df[f'TR_{p}'] = tr
        df[f'ATR_{p}'] = tr.rolling(window=p).mean()
    return df

def calculate_macd(close: PDSeries, fast: int=12, slow: int=26, signal: int=9) -> tuple[PDSeries, PDSeries]:
    """Return MACD and signal line series."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return (macd_line, signal_line)


def vwap(prices: np.ndarray, volumes: np.ndarray) -> float:
    """Return the volume weighted average price for prices with validation."""
    try:
        if len(prices) == 0 or len(volumes) == 0:
            raise ValueError('Prices and volumes arrays cannot be empty')
        if len(prices) != len(volumes):
            raise ValueError('Prices and volumes arrays must have same length')
        if np.isnan(prices).all() or np.isnan(volumes).all():
            raise ValueError('Input arrays contain only NaN values')
        total_volume: float = float(np.sum(volumes))
        if total_volume == 0:
            raise ValueError('Total volume cannot be zero')
        return float(np.sum(prices * volumes) / total_volume)
    except (ValueError, TypeError, ZeroDivisionError):
        return 0.0


def obv(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """Return On-Balance Volume (OBV) series (vectorized).

    Semantics are identical to the loop implementation:
    - If close[t] > close[t-1] ⇒ add volume[t]
    - If close[t] < close[t-1] ⇒ subtract volume[t]
    - If equal ⇒ carry forward
    The first OBV value is 0.
    """
    c = np.asarray(closes, dtype=float)
    v = np.asarray(volumes, dtype=float)
    if c.ndim != 1 or v.ndim != 1 or c.size != v.size:
        raise ValueError('closes and volumes must be same-length 1D arrays')
    n = c.size
    if n == 0:
        return cast(np.ndarray, np.array([], dtype=float))
    if n == 1:
        return cast(np.ndarray, np.array([0.0], dtype=float))
    delta = np.diff(c)
    sign = np.sign(delta)
    adj = sign * v[1:]
    obv_vals = np.concatenate([[0.0], np.cumsum(adj, dtype=float)])
    return cast(np.ndarray, obv_vals)


__all__ = [
    "IndicatorManager",
    "Indicator",
    "rsi_numba",
    "ema",
    "sma",
    "compute_sma",
    "compute_atr",
    "rsi",
    "atr",
    "mean_reversion_zscore",
    "calculate_macd",
    "vwap",
    "obv",
]
