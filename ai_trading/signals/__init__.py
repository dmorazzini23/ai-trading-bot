"""Simple signal generation module for tests."""

from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

import importlib
import hashlib
import logging
import math
import statistics
import time
from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import Any, TYPE_CHECKING, TypeAlias, cast

if TYPE_CHECKING:  # pragma: no cover - used for type hints
    import numpy as np  # type: ignore
    from ai_trading.utils.lazy_imports import load_pandas

    pd = load_pandas()

_ALPACA_PY_REQUIRED = (
    "alpaca-py==0.42.1 is required; install with `pip install alpaca-py==0.42.1`"
)

class _FallbackAPIError(Exception):
    """Placeholder used when Alpaca imports are intentionally unavailable."""


try:
    from alpaca.common.exceptions import APIError as _ImportedAPIError
except (ImportError, ModuleNotFoundError, AttributeError, OSError, RuntimeError):
    APIError: type[Exception] = _FallbackAPIError
else:
    APIError = _ImportedAPIError

from ai_trading.logging import get_logger
from ai_trading.utils import clamp_timeout as _clamp_timeout, clamp_request_timeout
from ai_trading.exc import RequestException

logger = get_logger(__name__)
_log = logger


def psleep(_=None) -> None:
    """Benchmark-friendly minimal sleep that accepts/ignores one arg."""
    time.sleep(0.01)


def clamp_timeout(df):
    """Benchmark-friendly helper returning input."""
    _ = _clamp_timeout(1.0)
    return df


from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ai_trading.utils import http
from ai_trading.utils.time import utcnow
from ai_trading.config import get_settings
from ai_trading.config.management import get_env as _get_env
from ai_trading.indicators import atr, mean_reversion_zscore, rsi
from ai_trading.utils.lazy_imports import load_pandas

# Heavy imports are loaded lazily within functions


pd = load_pandas()
_PANDAS_DATAFRAME_TYPE = pd.DataFrame

if TYPE_CHECKING:
    import pandas as _pd

    PDDataFrame: TypeAlias = _pd.DataFrame
    PDSeries: TypeAlias = _pd.Series
else:
    PDDataFrame = Any
    PDSeries = Any


def _get_numpy():
    try:  # pragma: no cover - import is tested indirectly
        import numpy as np  # type: ignore

        return np
    except AI_TRADING_FALLBACK_EXCEPTIONS:  # ImportError or others
        logger.debug("NUMPY_IMPORT_FAILED", exc_info=True)
        return None


def _get_pandas():
    try:
        if pd is None or not hasattr(pd, "DataFrame"):
            return None
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        logger.debug("PANDAS_HANDLE_CHECK_FAILED", exc_info=True)
        return None
    return pd


def _get_gaussian_hmm():
    try:  # pragma: no cover - import is tested indirectly
        from hmmlearn.hmm import GaussianHMM  # type: ignore

        return GaussianHMM
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        logger.debug("GAUSSIAN_HMM_IMPORT_FAILED", exc_info=True)
        return None


GaussianHMM = _get_gaussian_hmm()
_INDICATOR_CACHE_VERSION = "v2"


def rolling_mean(arr, window: int):
    """Simple rolling mean using cumulative sum for speed."""
    np = _get_numpy()
    if np is None:
        return []
    if window <= 0:
        raise ValueError("window must be positive")
    arr = np.asarray(arr, dtype=float)
    if arr.size < window:
        return np.array([], dtype=float)
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)


_portfolio_optimizer = None
_transaction_cost_calculator = None
_regime_detector = None
from ai_trading.position.intelligent_manager import (
    IntelligentPositionManager,
    PositionAction,
)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def generate() -> int:
    """Placeholder generate function used in tests."""
    return 0


def generate_signals(scored: PDDataFrame, buy_threshold: float) -> Any:
    pd = _get_pandas()
    if pd is None or not hasattr(pd, "DataFrame"):
        return scored
    try:
        signals = scored.copy()
        return signals
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        _log.error("SIGNAL_GEN_FAILED", extra={"cause": e.__class__.__name__, "detail": str(e)})
        raise


def _validate_macd_input(close_prices, min_len):
    np = _get_numpy()
    if np is None:
        return False
    has_nan = bool(close_prices.isna().any())
    try:
        values = np.asarray(close_prices, dtype=float)
    except (TypeError, ValueError):
        return False
    has_inf = bool(np.isinf(values).any())
    if has_nan or has_inf:
        return False
    if len(close_prices) < min_len:
        logger.warning(
            "MACD input length %s below minimum %s", len(close_prices), min_len
        )
        return False
    return True


def _compute_macd_df(close_prices, fast_period: int, slow_period: int, signal_period: int):
    """Compute MACD dataframe with graceful fallback."""
    pd = _get_pandas()
    if pd is None or not hasattr(pd, "DataFrame"):
        return None
    fast_ema = close_prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close_prices.ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    try:
        return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": histogram})
    except AttributeError:
        return None


@lru_cache(maxsize=128)
def _cached_macd(prices_tuple: tuple, fast_period: int, slow_period: int, signal_period: int) -> Any | None:
    pd = _get_pandas()
    if pd is None or not hasattr(pd, "Series"):
        return None
    try:
        series = pd.Series(prices_tuple)
    except AttributeError:
        return None
    return _compute_macd_df(series, fast_period, slow_period, signal_period)


def calculate_macd(close_prices, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Any | None:
    """Calculate MACD indicator values with validation.

    Parameters
    ----------
    close_prices
        Series of closing prices.
    fast_period : int
        Fast EMA period. Defaults to ``12``.
    slow_period : int
        Slow EMA period. Defaults to ``26``.
    signal_period : int
        Signal line EMA period. Defaults to ``9``.

    Returns
    -------
    Optional[Any]
        DataFrame containing ``macd``, ``signal`` and ``histogram`` columns or
        ``None`` if the calculation fails.
    """
    pd = _get_pandas()
    if pd is None or not hasattr(pd, "DataFrame"):
        logger.warning("Pandas not available, cannot calculate MACD")
        return None
    try:
        # Allow callers to pass a full OHLCV frame; MACD only uses close prices.
        if hasattr(close_prices, "columns"):
            columns = getattr(close_prices, "columns", [])
            if "close" in columns:
                close_prices = close_prices["close"]
            elif len(columns) == 1:
                close_prices = close_prices.iloc[:, 0]
            else:
                return None
        min_len = slow_period + signal_period
        if not _validate_macd_input(close_prices, min_len):
            return None
        tup = tuple(map(float, close_prices.dropna().tolist()))
        macd_df = _cached_macd(tup, fast_period, slow_period, signal_period)
        if macd_df is None or macd_df.isnull().values.any():
            logger.warning("MACD calculation returned NaNs in the result")
            return None
        return macd_df
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        logger.error("MACD_CALCULATION_FAILED", exc_info=True, extra={"cause": e.__class__.__name__, "detail": str(e)})
        return None


def _validate_input_df(data) -> None:
    pd = _get_pandas()
    if data is None:
        raise ValueError("Input must be a DataFrame")
    if pd is not None and hasattr(pd, "DataFrame") and (not isinstance(data, _PANDAS_DATAFRAME_TYPE)):
        raise ValueError("Input must be a DataFrame")
    required = ["open", "high", "low", "close", "volume"]
    if hasattr(data, "columns"):
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise KeyError(f"missing required column(s): {missing}")


def _apply_macd(data) -> Any | None:
    pd = _get_pandas()
    if pd is None or not hasattr(pd, "DataFrame"):
        logger.warning("Pandas not available for MACD application")
        return None
    macd_df = calculate_macd(data["close"])
    if macd_df is None or macd_df.empty:
        logger.warning("MACD indicator calculation failed, returning None")
        raise ValueError("MACD calculation failed")
    missing = [c for c in ("macd", "signal", "histogram") if c not in macd_df]
    if missing:
        raise ValueError(f"MACD output missing column(s) {missing}")
    data[["macd", "signal", "histogram"]] = macd_df[["macd", "signal", "histogram"]].astype(float)
    logger.debug(f"After MACD {(data.columns[0] if not data.empty else '')}, tail close:\n{data[['close']].tail(5)}")
    return data


def _apply_psar(data) -> Any:
    pd = _get_pandas()
    if pd is None or not hasattr(pd, "DataFrame"):
        logger.warning("Pandas not available for PSAR application")
        return data
    from .indicators import psar as _psar

    return _psar(data)


def _indicator_cache_path(ticker: str, data) -> Path:
    """Return a cache path tied to the input data and indicator version."""
    safe_ticker = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in ticker)
    fingerprint = _fingerprint_indicator_data(data)
    return Path(f"cache_{safe_ticker}_{_INDICATOR_CACHE_VERSION}_{fingerprint}.parquet")


def _fingerprint_indicator_data(data) -> str:
    pd = _get_pandas()
    digest = hashlib.blake2b(digest_size=12)
    digest.update(_INDICATOR_CACHE_VERSION.encode("ascii"))
    try:
        cols = [col for col in ("open", "high", "low", "close", "volume") if col in data.columns]
        digest.update("|".join(cols).encode("utf-8"))
        if pd is not None and hasattr(pd, "util"):
            hashed = pd.util.hash_pandas_object(data[cols], index=True).values
            digest.update(hashed.tobytes())
        else:
            digest.update(str(data[cols].to_dict("split")).encode("utf-8"))
    except (AttributeError, KeyError, TypeError, ValueError):
        digest.update(str(len(data) if hasattr(data, "__len__") else "unknown").encode("utf-8"))
        digest.update(str(getattr(data, "index", "")).encode("utf-8"))
    return digest.hexdigest()


def prepare_indicators(data, ticker: str | None = None) -> Any | None:
    """Prepare indicator columns for a trading strategy.

    Parameters
    ----------
    data
        Market data containing ``open``, ``high``, ``low``, ``close`` and
        ``volume`` columns.

    Returns
    -------
    Any
        Data enriched with indicator columns.

    Raises
    ------
    KeyError
        If required OHLCV columns are missing from ``data``.
    ValueError
        If the MACD indicator fails to calculate.
    """
    pd = _get_pandas()
    _validate_input_df(data)
    cache_path = _indicator_cache_path(ticker, data) if ticker else None
    if _get_env("DISABLE_PARQUET", False, cast=bool, resolve_aliases=False):
        cache_path = None
    if (
        pd is not None
        and hasattr(pd, "DataFrame")
        and cache_path
        and cache_path.exists()
    ):
        try:
            return pd.read_parquet(cache_path)
        except (AttributeError, ImportError, OSError, ValueError) as e:
            logger.warning(
                "INDICATOR_CACHE_READ_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e), "path": str(cache_path)},
            )
    data = _apply_macd(data.copy())
    data = _apply_psar(data)
    if pd is not None:
        logger.debug(f"After prepare_macd {ticker or ''}, tail close:\n{data[['close']].tail(5)}")
    if pd is not None and cache_path:
        try:
            data.to_parquet(cache_path, engine="pyarrow")
        except (AttributeError, ImportError, OSError, ValueError) as e:
            logger.warning(
                "INDICATOR_CACHE_WRITE_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e), "path": str(cache_path)},
            )
    return data


def generate_position_hold_signals(ctx, current_positions: list) -> dict:
    """Generate hold signals for existing positions to reduce churn."""
    try:
        if IntelligentPositionManager is None:
            logger.warning("IntelligentPositionManager not available - no hold signals generated")
            return {}
        if not hasattr(ctx, "position_manager"):
            ctx.position_manager = IntelligentPositionManager(ctx)
        hold_signals: dict[str, str] = {}
        for position in current_positions:
            symbol = getattr(position, "symbol", "")
            if not symbol:
                continue
            rec = ctx.position_manager.analyze_position(symbol, position, current_positions)
            if rec.action == PositionAction.HOLD:
                hold_signals[symbol] = "hold"
            elif rec.action in (
                PositionAction.FULL_SELL,
                PositionAction.PARTIAL_SELL,
                PositionAction.REDUCE_SIZE,
            ):
                hold_signals[symbol] = "sell"
            else:
                hold_signals[symbol] = "neutral"
        logger.info(
            "POSITION_HOLD_SIGNALS_GENERATED",
            extra={
                "signals_count": len(hold_signals),
                "hold_count": sum(1 for s in hold_signals.values() if s == "hold"),
                "sell_count": sum(1 for s in hold_signals.values() if s == "sell"),
                "neutral_count": sum(1 for s in hold_signals.values() if s == "neutral"),
            },
        )
        return hold_signals
    except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        logger.error(
            "SIGNAL_PROCESSING_FAILED",
            extra={"cause": exc.__class__.__name__, "detail": str(exc)},
        )
        return {}


def enhance_signals_with_position_logic(signals: list, ctx, hold_signals: dict = None) -> list:
    """Enhance trading signals with position holding logic."""
    try:
        if hold_signals is None:
            hold_signals = {}
        enhanced_signals = []
        for signal in signals:
            symbol = getattr(signal, "symbol", "")
            side = getattr(signal, "side", "")
            if not symbol:
                enhanced_signals.append(signal)
                continue
            if symbol in hold_signals:
                hold_action = hold_signals[symbol]
                if hold_action == "hold" and side == "sell":
                    logger.info("SIGNAL_CONVERTED_HOLD | %s sell->hold", symbol)
                    continue
                elif hold_action == "sell" and side == "buy":
                    logger.info("SIGNAL_SKIP_BUY_SELL_PENDING | %s buy skipped, sell pending", symbol)
                    continue
            enhanced_signals.append(signal)
        logger.info(
            "SIGNALS_ENHANCED",
            extra={
                "original_count": len(signals),
                "enhanced_count": len(enhanced_signals),
                "filtered_count": len(signals) - len(enhanced_signals),
            },
        )
        return enhanced_signals
    except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        logger.error("SIGNAL_PROCESSING_FAILED", extra={"cause": exc.__class__.__name__, "detail": str(exc)})
        return signals

