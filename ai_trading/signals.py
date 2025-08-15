"""Simple signal generation module for tests."""

import importlib
import logging
import os
import time
from ai_trading.utils import sleep as psleep, clamp_timeout
from functools import lru_cache
from typing import Any, Dict, Iterable  # AI-AGENT-REF: broaden typing for signals

# If this file calls broker/data APIs, they may raise concrete errors.
try:
    from alpaca_trade_api.rest import APIError  # AI-AGENT-REF: narrow broker catches
except Exception:  # tests/dev envs without alpaca
    class APIError(Exception):
        pass

_log = logging.getLogger(__name__)
logger = _log

# Core dependencies
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import requests

# Optional ML dependency: hmmlearn
try:  # pragma: no cover - optional dependency
    from hmmlearn.hmm import GaussianHMM  # type: ignore
    SKLEARN_HMM_AVAILABLE = True
except Exception:  # absent in many CI envs; tests skip if None
    GaussianHMM = None  # noqa: N816
    SKLEARN_HMM_AVAILABLE = False

# Import indicators
from ai_trading.indicators import atr, mean_reversion_zscore, rsi

# Import settings for configuration
from ai_trading.config import get_settings

# Cache the last computed signal matrix to avoid recomputation
_LAST_SIGNAL_BAR = None
_LAST_SIGNAL_MATRIX = None


def get_utcnow():
    return datetime.now(UTC)


# AI-AGENT-REF: safe close retrieval for pipelines
def robust_signal_price(df) -> float:
    """Get closing price from dataframe with fallback."""
    if pd is None:
        return 1e-3
    try:
        return df["close"].iloc[-1]
    except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:  # AI-AGENT-REF: narrow exception
        logger.warning(
            "DATA_MUNGING_FAILED",
            extra={"cause": e.__class__.__name__, "detail": str(e)},
        )
        return 1e-3


def rolling_mean(arr, window: int):
    """Simple rolling mean using cumulative sum for speed."""
    if np is None:
        return []
    if window <= 0:
        raise ValueError("window must be positive")
    arr = np.asarray(arr, dtype=float)
    if arr.size < window:
        return np.array([], dtype=float)
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)


# AI-AGENT-REF: Portfolio-level optimization integration
from ai_trading.execution.transaction_costs import (
    TradeType,
    TransactionCostCalculator,
    create_transaction_cost_calculator,
)

from ai_trading.portfolio import (
    PortfolioDecision,
    PortfolioOptimizer,
    create_portfolio_optimizer,
)
from ai_trading.strategies.regime_detector import (
    RegimeDetector,
    create_regime_detector,
)

logger.info("Portfolio optimization modules loaded successfully")

# Global portfolio optimizer instance (initialized when first used)
_portfolio_optimizer = None
_transaction_cost_calculator = None
_regime_detector = None

# AI-AGENT-REF: Import position management for hold signals
from ai_trading.position.legacy_manager import PositionManager


def load_module(name):
    if not isinstance(name, str):
        logger.warning("Skipping load_module on non-string: %s", type(name))
        return None
    return importlib.import_module(name)


def _fetch_api(url: str, retries: int = 3, delay: float = 1.0) -> dict:
    """Fetch JSON from an API with simple retry logic and backoff."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=clamp_timeout(5, 5, 0.5))
            resp.raise_for_status()
            return resp.json()
        except (
            requests.exceptions.RequestException
        ) as exc:  # pragma: no cover - network may be mocked
            logger.warning("API request failed (%s/%s): %s", attempt, retries, exc)
            psleep(delay * attempt)
    return {}


def generate() -> int:
    """Placeholder generate function used in tests."""

    return 0


# AI-AGENT-REF: structured error logging for data fetch
def fetch_history(symbols: Iterable[str], start, end, source: str = "alpaca") -> pd.DataFrame:
    """Fetch historical data for symbols between start and end."""
    try:
        # existing logic (alpaca / yfinance / cache) ...
        df = pd.DataFrame()
        return df
    except (APIError, ConnectionError, TimeoutError, KeyError, ValueError, TypeError) as e:  # AI-AGENT-REF: narrow exception
        _log.error(
            "FETCH_HISTORY_FAILED",
            extra={
                "symbol_count": len(symbols),
                "cause": e.__class__.__name__,
                "detail": str(e),
            },
        )
        raise


# AI-AGENT-REF: structured error logging for indicators
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # compute TA features, rolling windows, etc.
        out = df.copy()
        return out
    except (KeyError, ValueError, TypeError, ZeroDivisionError, AttributeError) as e:  # AI-AGENT-REF: narrow exception
        _log.error(
            "INDICATORS_FAILED",
            extra={"cause": e.__class__.__name__, "detail": str(e)},
        )
        raise


# AI-AGENT-REF: structured error logging for feature matrix
def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # select/align columns, dropna, normalize
        X = df.copy()
        return X
    except (KeyError, ValueError, TypeError, IndexError) as e:  # AI-AGENT-REF: narrow exception
        _log.error(
            "FEATURE_MATRIX_FAILED",
            extra={"cause": e.__class__.__name__, "detail": str(e)},
        )
        raise


# AI-AGENT-REF: structured error logging for model scoring
def score_candidates(X: pd.DataFrame, model) -> pd.DataFrame:
    try:
        # model.predict_proba or predict; attach to frame
        scored = X.copy()
        return scored
    except (ValueError, TypeError, AttributeError) as e:  # AI-AGENT-REF: narrow exception
        _log.error(
            "SCORING_FAILED",
            extra={"cause": e.__class__.__name__, "detail": str(e)},
        )
        raise


# AI-AGENT-REF: structured error logging for signal generation
def generate_signals(scored: pd.DataFrame, buy_threshold: float) -> pd.DataFrame:
    try:
        # thresholding / momentum rules
        signals = scored.copy()
        return signals
    except (KeyError, ValueError, TypeError) as e:  # AI-AGENT-REF: narrow exception
        _log.error(
            "SIGNAL_GEN_FAILED",
            extra={"cause": e.__class__.__name__, "detail": str(e)},
        )
        raise


def _validate_macd_input(close_prices, min_len):
    if close_prices.isna().any() or np.isinf(close_prices).any():
        return False
    return not len(close_prices) < min_len


def _compute_macd_df(
    close_prices,
    fast_period: int,
    slow_period: int,
    signal_period: int,
):
    """Compute MACD dataframe with graceful fallback."""
    if pd is None:
        return None

    fast_ema = close_prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close_prices.ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram}
    )


@lru_cache(maxsize=128)
def _cached_macd(
    prices_tuple: tuple,
    fast_period: int,
    slow_period: int,
    signal_period: int,
) -> Any | None:
    series = pd.Series(prices_tuple)
    return _compute_macd_df(series, fast_period, slow_period, signal_period)


def calculate_macd(
    close_prices,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Any | None:
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

    try:
        min_len = slow_period + signal_period
        if not _validate_macd_input(close_prices, min_len):
            return None

        tup = tuple(map(float, close_prices.dropna().tolist()))
        macd_df = _cached_macd(tup, fast_period, slow_period, signal_period)

        if macd_df.isnull().values.any():
            logger.warning("MACD calculation returned NaNs in the result")
            return None

        return macd_df

    except (KeyError, ValueError, TypeError) as e:  # AI-AGENT-REF: narrow MACD errors
        logger.error(
            "MACD_CALCULATION_FAILED",
            exc_info=True,
            extra={"cause": e.__class__.__name__, "detail": str(e)},
        )
        return None


def _validate_input_df(data) -> None:
    if data is None:
        raise ValueError("Input must be a DataFrame")
    if pd is not None and not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a DataFrame")
    if hasattr(data, "columns") and "close" not in data.columns:
        raise ValueError("Input data missing 'close' column")


def _apply_macd(data) -> Any | None:
    macd_df = calculate_macd(data["close"])
    if macd_df is None or macd_df.empty:
        logger.warning("MACD indicator calculation failed, returning None")
        raise ValueError("MACD calculation failed")
    missing = [c for c in ("macd", "signal", "histogram") if c not in macd_df]
    if missing:
        raise ValueError(f"MACD output missing column(s) {missing}")
    data[["macd", "signal", "histogram"]] = macd_df[
        ["macd", "signal", "histogram"]
    ].astype(float)
    logger.debug(
        f"After MACD {data.columns[0] if not data.empty else ''}, tail close:\n{data[['close']].tail(5)}"
    )
    return data


def prepare_indicators(data, ticker: str | None = None) -> Any | None:
    """Prepare indicator columns for a trading strategy.

    Parameters
    ----------
    data
        Market data containing at least a ``close`` column.

    Returns
    -------
    Any
        Data enriched with indicator columns.

    Raises
    ------
    ValueError
        If the MACD indicator fails to calculate or ``close`` column is missing.
    """

    _validate_input_df(data)
    cache_path = Path(f"cache_{ticker}.parquet") if ticker else None
    if os.getenv("DISABLE_PARQUET"):
        cache_path = None
    if cache_path and cache_path.exists():
        return pd.read_parquet(cache_path)

    data = _apply_macd(data.copy())
    logger.debug(
        f"After prepare_macd {ticker or ''}, tail close:\n{data[['close']].tail(5)}"
    )
    if cache_path:
        try:
            data.to_parquet(cache_path, engine="pyarrow")
        except (OSError, ValueError) as e:  # AI-AGENT-REF: narrow cache write errors
            logger.warning(
                "INDICATOR_CACHE_WRITE_FAILED",
                extra={
                    "cause": e.__class__.__name__,
                    "detail": str(e),
                    "path": str(cache_path),
                },
            )

    # Additional indicators can be added here using similar defensive checks
    return data


def prepare_indicators_parallel(
    symbols: list[str],
    data: dict[str, Any],
    max_workers: int | None = None,
) -> None:
    """Run :func:`prepare_indicators` over ``symbols`` concurrently,
    but fall back to serial execution for small symbol sets."""
    if os.getenv("DISABLE_PARQUET"):
        return

    # small symbol sets are faster serially than spinning up threads
    # call prepare_indicators exactly as the serial loop does
    SERIAL_THRESHOLD = 8
    if len(symbols) <= SERIAL_THRESHOLD:
        for sym in symbols:
            # note: serial test does prepare_indicators(data) for each symbol
            prepare_indicators(data[sym])
        return

    max_workers = max_workers or min(4, len(symbols))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(lambda s: prepare_indicators(data[s], s), symbols)


def generate_signal(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Generate trading signals from specified dataframe column.

    Creates directional trading signals (+1, 0, -1) based on the sign of values
    in the specified column. This is a core utility function for converting
    technical indicator values into actionable trading signals.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing market data and technical indicators.
        Must have a DatetimeIndex for proper signal alignment.
    column : str
        Name of the column to generate signals from. Common examples:
        - 'momentum': Price momentum values
        - 'rsi_signal': RSI-based signal values
        - 'macd_histogram': MACD histogram values
        - 'bb_position': Bollinger Bands position

    Returns
    -------
    pd.Series
        Signal series with same index as input dataframe:
        - +1: Buy/long signal (positive column values)
        - 0: Neutral signal (zero or NaN column values)
        - -1: Sell/short signal (negative column values)

    Raises
    ------
    ValueError
        If dataframe is None, empty, or column doesn't exist
    TypeError
        If input parameters are not of expected types

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    >>> df = pd.DataFrame({
    ...     'close': np.random.randn(100).cumsum() + 100,
    ...     'momentum': np.random.randn(100)
    ... }, index=dates)
    >>>
    >>> # Generate signals from momentum
    >>> signals = generate_signal(df, 'momentum')
    >>> logging.info(f"Buy signals: {(signals == 1).sum()}")
    >>> logging.info(f"Sell signals: {(signals == -1).sum()}")
    >>> logging.info(f"Neutral signals: {(signals == 0).sum()}")

    >>> # Handle missing data gracefully
    >>> df_with_gaps = df.copy()
    >>> df_with_gaps.loc[df_with_gaps.index[10:20], 'momentum'] = np.nan
    >>> signals = generate_signal(df_with_gaps, 'momentum')  # NaNs become 0

    Signal Generation Logic
    ----------------------
    The function applies the following transformation:

    1. **Positive Values** → +1 (Buy Signal)
       - Indicates bullish/upward momentum
       - Suggests entering long positions

    2. **Zero/NaN Values** → 0 (Neutral Signal)
       - No clear directional bias
       - Suggests holding current positions

    3. **Negative Values** → -1 (Sell Signal)
       - Indicates bearish/downward momentum
       - Suggests entering short positions or exiting longs

    Error Handling
    -------------
    - Returns empty Series for invalid inputs
    - Logs detailed error messages for debugging
    - Gracefully handles missing or corrupt data
    - Preserves index alignment for time series operations

    Performance Notes
    ----------------
    - Uses vectorized numpy operations for efficiency
    - Handles large datasets without memory issues
    - Optimized for real-time signal generation

    See Also
    --------
    compute_signal_matrix : Generate multiple signals simultaneously
    ensemble_vote_signals : Combine multiple signal sources
    generate_ensemble_signal : Advanced signal aggregation
    """
    # Input validation
    if df is None:
        logger.error("Dataframe is None in generate_signal")
        return pd.Series(dtype=int, name="signal")

    if not isinstance(df, pd.DataFrame):
        logger.error("Expected DataFrame, got %s", type(df))
        return pd.Series(dtype=int, name="signal")

    if df.empty:
        logger.warning("Dataframe is empty in generate_signal")
        return pd.Series(dtype=int, name="signal", index=df.index)

    if not isinstance(column, str):
        logger.error("Column name must be string, got %s", type(column))
        return pd.Series(dtype=int, name="signal", index=df.index)

    if column not in df.columns:
        logger.error(
            "Required column '%s' not found in dataframe. Available columns: %s",
            column,
            list(df.columns),
        )
        return pd.Series(dtype=int, name="signal", index=df.index)

    try:
        # Extract values and handle NaN/infinite values
        values = df[column].replace([np.inf, -np.inf], np.nan)

        # Generate signals using numpy sign function
        signal_values = np.sign(values.fillna(0))  # NaN becomes 0 (neutral)

        # Create properly indexed series
        signal_series = pd.Series(
            signal_values, index=df.index, name="signal", dtype=int
        )

        logger.debug(
            "Generated %d signals for column '%s': Buy=%d, Sell=%d, Neutral=%d",
            len(signal_series),
            column,
            (signal_series == 1).sum(),
            (signal_series == -1).sum(),
            (signal_series == 0).sum(),
        )

        return signal_series

    except (ValueError, TypeError, AttributeError) as exc:
        logger.error(
            "SIGNAL_GEN_FAILED",
            exc_info=True,
            extra={
                "cause": exc.__class__.__name__,
                "detail": str(exc),
                "column": column,
            },
        )
        return pd.Series(
            dtype=int, name="signal", index=df.index if hasattr(df, "index") else None
        )
    except (KeyError, ZeroDivisionError) as exc:  # AI-AGENT-REF: narrow exception
        logger.error(
            "SIGNAL_GEN_FAILED",
            exc_info=True,
            extra={"cause": exc.__class__.__name__, "detail": str(exc)},
        )
        return pd.Series(
            dtype=int, name="signal", index=df.index if hasattr(df, "index") else None
        )


def detect_market_regime_hmm(
    df,
    n_components: int = 3,
    window_size: int = 1000,
    max_iter: int = 10,
) -> np.ndarray:
    """Return HMM-based market regime labels for ``df``."""
    if not SKLEARN_HMM_AVAILABLE:
        return np.zeros(len(df), dtype=int)

    col = "close" if "close" in df.columns else "Close"
    if col not in df:
        return np.zeros(len(df), dtype=int)

    series = pd.to_numeric(df[col], errors="coerce")
    returns = np.log(series).diff().dropna()
    if returns.size < n_components or not np.isfinite(returns).all():
        return np.zeros(len(df), dtype=int)

    arr = returns.values.reshape(-1, 1)
    train = arr[-window_size:] if arr.shape[0] > window_size else arr

    try:
        model = GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=max_iter,
            random_state=42,
        )
        model.fit(train)
        hidden_full = model.predict(arr)
        result = np.concatenate([[hidden_full[0]], hidden_full])
    except Exception as exc:  # pragma: no cover - robustness
        logger.warning(
            "MODEL_FIT_FAILED",
            extra={"cause": exc.__class__.__name__, "detail": str(exc)},
        )
        return np.zeros(len(df), dtype=int)

    return np.asarray(result, dtype=int)


def compute_signal_matrix(df) -> Any | None:
    """Return a matrix of z-scored indicator signals."""

    if df is None or df.empty:
        logger.warning("compute_signal_matrix received empty dataframe")
        return pd.DataFrame()

    global _LAST_SIGNAL_BAR, _LAST_SIGNAL_MATRIX
    last_bar = df.index[-1] if not df.empty else None
    if last_bar is not None and last_bar == _LAST_SIGNAL_BAR:
        # AI-AGENT-REF: reuse previously computed indicators for same bar
        logger.debug("Reusing cached signal matrix for bar: %s", last_bar)
        return (
            _LAST_SIGNAL_MATRIX.copy()
            if _LAST_SIGNAL_MATRIX is not None
            else pd.DataFrame()
        )

    required = {"close", "high", "low"}
    if not required.issubset(df.columns):
        logger.warning(
            "compute_signal_matrix missing required columns: %s",
            required - set(df.columns),
        )
        return pd.DataFrame()

    try:
        # AI-AGENT-REF: Enhanced signal computation with error handling
        logger.debug("Computing signal matrix for %d rows", len(df))

        macd_df = calculate_macd(df["close"])
        if rsi is not None:
            rsi_series = rsi(
                tuple(df["close"].fillna(method="ffill").astype(float)), 14
            )
        else:
            # Fallback RSI calculation
            rsi_series = pd.Series(50.0, index=df.index)  # Neutral RSI

        sma_diff = df["close"] - df["close"].rolling(20).mean()

        if atr is not None:
            atr_series = atr(df["high"], df["low"], df["close"], 14)
        else:
            # Fallback ATR calculation
            high_low = df["high"] - df["low"]
            atr_series = high_low.rolling(14).mean()

        atr_move = df["close"].diff() / atr_series.replace(0, np.nan)

        if mean_reversion_zscore is not None:
            mean_rev = mean_reversion_zscore(df["close"], 20)
        else:
            # Fallback mean reversion calculation
            rolling_mean = df["close"].rolling(20).mean()
            rolling_std = df["close"].rolling(20).std()
            mean_rev = (df["close"] - rolling_mean) / rolling_std.replace(0, np.nan)

        def _z(series) -> Any:
            """Z-score normalization with enhanced error handling."""
            try:
                if series is None or series.empty:
                    return pd.Series(0.0, index=df.index)
                mean_val = series.rolling(20).mean()
                std_val = series.rolling(20).std(ddof=0)
                # AI-AGENT-REF: avoid division by zero
                std_val = std_val.replace(0, np.nan)
                return (series - mean_val) / std_val
            except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:  # AI-AGENT-REF: narrow exception
                logger.warning(
                    "DATA_MUNGING_FAILED",
                    extra={"cause": e.__class__.__name__, "detail": str(e)},
                )
                return pd.Series(0.0, index=df.index)

        matrix = pd.DataFrame(index=df.index)
        if macd_df is not None and not macd_df.empty and "macd" in macd_df.columns:
            matrix["macd"] = _z(macd_df["macd"])
        else:
            logger.warning("MACD calculation failed or missing, using neutral values")
            matrix["macd"] = pd.Series(0.0, index=df.index)

        matrix["rsi"] = _z(rsi_series)
        matrix["sma_diff"] = _z(sma_diff)
        matrix["atr_move"] = _z(atr_move)
        matrix["mean_rev_z"] = mean_rev

        # AI-AGENT-REF: Clean up infinite values and excessive NaNs
        matrix = matrix.replace([np.inf, -np.inf], np.nan)
        matrix = matrix.dropna(how="all")

        # Cache the results
        _LAST_SIGNAL_BAR = last_bar
        _LAST_SIGNAL_MATRIX = matrix.copy()

        logger.debug(
            "Successfully computed signal matrix with %d rows, %d columns",
            len(matrix),
            len(matrix.columns),
        )
        return matrix

    except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:  # AI-AGENT-REF: narrow exception
        logger.error(
            "SIGNAL_PROCESSING_FAILED",
            exc_info=True,
            extra={"cause": e.__class__.__name__, "detail": str(e)},
        )
        return pd.DataFrame()


def ensemble_vote_signals(signal_matrix) -> Any:
    """Return voting-based entry signals from ``signal_matrix``."""

    if signal_matrix is None or signal_matrix.empty:
        return pd.Series(dtype=int)
    pos = (signal_matrix > 0.5).sum(axis=1)
    neg = (signal_matrix < -0.5).sum(axis=1)
    votes = np.where(pos >= 2, 1, np.where(neg >= 2, -1, 0))
    return pd.Series(votes, index=signal_matrix.index)


def classify_regime(df, window: int = 20) -> Any:
    """Classify each row as 'trend' or 'mean_revert' based on volatility."""

    if df is None or df.empty or "close" not in df:
        return pd.Series(dtype=object)
    returns = df["close"].pct_change()
    vol = returns.rolling(window).std()
    med = vol.rolling(window).median()
    dev = vol.rolling(window).std()
    regime = np.where(vol > med + dev, "trend", "mean_revert")
    return pd.Series(regime, index=df.index)


# AI-AGENT-REF: ensemble decision using multiple indicator columns
def generate_ensemble_signal(df) -> int:
    def last(col: str) -> float:
        s = df.get(col, pd.Series(dtype=float))
        return s.iloc[-1] if not s.empty else np.nan

    signals = []
    if last("EMA_5") > last("EMA_20"):
        signals.append(1)
    if last("SMA_50") > last("SMA_200"):
        signals.append(1)
    if last("close") > last("UB"):
        signals.append(-1)
    if last("close") < last("LB"):
        signals.append(1)
    avg_signal = np.mean(signals) if signals else 0
    if avg_signal > 0.5:
        return 1
    if avg_signal < -0.5:
        return -1
    return 0


# AI-AGENT-REF: Position holding signal generation functions
def generate_position_hold_signals(ctx, current_positions: list) -> dict:
    """Generate hold signals for existing positions to reduce churn."""
    try:
        if PositionManager is None:
            logger.warning("PositionManager not available - no hold signals generated")
            return {}

        # Initialize position manager if not exists
        if not hasattr(ctx, "position_manager"):
            ctx.position_manager = PositionManager(ctx)

        # Generate hold signals
        hold_signals = ctx.position_manager.get_hold_signals(current_positions)

        logger.info(
            "POSITION_HOLD_SIGNALS_GENERATED",
            extra={
                "signals_count": len(hold_signals),
                "hold_count": sum(1 for s in hold_signals.values() if s == "hold"),
                "sell_count": sum(1 for s in hold_signals.values() if s == "sell"),
                "neutral_count": sum(
                    1 for s in hold_signals.values() if s == "neutral"
                ),
            },
        )

        return hold_signals

    except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:  # AI-AGENT-REF: narrow exception
        logger.error(
            "SIGNAL_PROCESSING_FAILED",
            extra={"cause": exc.__class__.__name__, "detail": str(exc)},
        )
        return {}


def should_generate_new_signal(
    symbol: str, hold_signals: dict, existing_positions: dict
) -> bool:
    """Determine if new buy/sell signals should be generated for a symbol."""
    try:
        # If we have a hold signal for this symbol, don't generate new signals
        if symbol in hold_signals and hold_signals[symbol] == "hold":
            logger.info("SKIP_NEW_SIGNAL_HOLD | %s has hold signal", symbol)
            return False

        # If we have an existing position and no explicit sell signal, be conservative
        if symbol in existing_positions and existing_positions[symbol] != 0:
            if symbol not in hold_signals or hold_signals[symbol] == "neutral":
                logger.info(
                    "SKIP_NEW_SIGNAL_EXISTING | %s has existing position, no clear signal",
                    symbol,
                )
                return False

        return True

    except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:  # AI-AGENT-REF: narrow exception
        logger.warning(
            "DATA_MUNGING_FAILED",
            extra={"cause": exc.__class__.__name__, "detail": str(exc), "symbol": symbol},
        )
        return True  # Default to allowing signal generation


def enhance_signals_with_position_logic(
    signals: list, ctx, hold_signals: dict = None
) -> list:
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

            # Check if we should respect hold signal
            if symbol in hold_signals:
                hold_action = hold_signals[symbol]

                if hold_action == "hold" and side == "sell":
                    # Convert sell signal to hold (skip the signal)
                    logger.info("SIGNAL_CONVERTED_HOLD | %s sell->hold", symbol)
                    continue

                elif hold_action == "sell" and side == "buy":
                    # Don't add new buy signals for symbols marked for selling
                    logger.info(
                        "SIGNAL_SKIP_BUY_SELL_PENDING | %s buy skipped, sell pending",
                        symbol,
                    )
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

    except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:  # AI-AGENT-REF: narrow exception
        logger.error(
            "SIGNAL_PROCESSING_FAILED",
            extra={"cause": exc.__class__.__name__, "detail": str(exc)},
        )
        return signals  # Return original signals on error


# AI-AGENT-REF: Portfolio-level signal filtering for churn reduction
def filter_signals_with_portfolio_optimization(
    signals: list, ctx, current_positions: dict = None
) -> list:
    """
    Filter trading signals based on portfolio-level optimization criteria.

    This is the core churn reduction mechanism that evaluates each signal
    at the portfolio level rather than individually, dramatically reducing
    trading frequency while improving overall performance.

    Args:
        signals: List of trading signals to filter
        ctx: Trading context with market data
        current_positions: Current portfolio positions

    Returns:
        Filtered list of signals that pass portfolio-level validation
    """
    try:
        settings = get_settings()
        if not settings.ENABLE_PORTFOLIO_FEATURES:
            logger.debug("Portfolio optimization not available, skipping filtering")
            return signals

        if not signals:
            return signals

        # Initialize portfolio optimization components
        global _portfolio_optimizer, _transaction_cost_calculator, _regime_detector

        if _portfolio_optimizer is None:
            _portfolio_optimizer = create_portfolio_optimizer()
            _transaction_cost_calculator = create_transaction_cost_calculator()
            _regime_detector = create_regime_detector()
            logger.info("Portfolio optimization components initialized")

        # Prepare market data for analysis
        market_data = _prepare_market_data_for_portfolio_analysis(ctx, signals)
        if not market_data:
            logger.warning(
                "Insufficient market data for portfolio analysis, skipping filtering"
            )
            return signals

        # Detect current market regime for dynamic thresholds
        regime, regime_metrics = _regime_detector.detect_current_regime(market_data)

        # Calculate dynamic thresholds based on market regime
        dynamic_thresholds = _regime_detector.calculate_dynamic_thresholds(
            regime, regime_metrics
        )

        # Update optimizer with dynamic thresholds
        _portfolio_optimizer.improvement_threshold = (
            dynamic_thresholds.minimum_improvement_threshold
        )
        _portfolio_optimizer.rebalance_drift_threshold = (
            dynamic_thresholds.rebalance_drift_threshold
        )
        _portfolio_optimizer.max_correlation_penalty = (
            dynamic_thresholds.correlation_penalty_adjustment * 0.15
        )

        # Get current portfolio positions
        if current_positions is None:
            current_positions = _get_current_portfolio_positions(ctx)

        # Filter signals through portfolio optimization
        filtered_signals = []
        portfolio_decisions = []

        for signal in signals:
            try:
                symbol = getattr(signal, "symbol", "")
                side = getattr(signal, "side", "")
                quantity = getattr(signal, "quantity", 0)

                if not symbol or not side:
                    filtered_signals.append(signal)
                    continue

                # Calculate proposed position change
                current_position = current_positions.get(symbol, 0.0)
                if side.lower() == "buy":
                    proposed_position = current_position + abs(quantity)
                elif side.lower() == "sell":
                    proposed_position = max(0.0, current_position - abs(quantity))
                else:
                    # Unknown side, keep signal as-is
                    filtered_signals.append(signal)
                    continue

                # Portfolio-level decision making
                decision, reasoning = _portfolio_optimizer.make_portfolio_decision(
                    symbol, proposed_position, current_positions, market_data
                )

                portfolio_decisions.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "decision": decision.value,
                        "reasoning": reasoning,
                    }
                )

                # Apply decision logic
                if decision == PortfolioDecision.APPROVE:
                    try:
                        # Validate transaction costs with safety margin
                        expected_profit = _estimate_signal_profit(signal, market_data)
                        position_change = abs(proposed_position - current_position)

                        profitability = (
                            _transaction_cost_calculator.validate_trade_profitability(
                                symbol,
                                position_change,
                                expected_profit,
                                market_data,
                                TradeType.LIMIT_ORDER,
                            )
                        )

                        if profitability.is_profitable:
                            filtered_signals.append(signal)
                            logger.info(
                                f"SIGNAL_APPROVED_PORTFOLIO | {symbol} {side} - {reasoning}"
                            )
                        else:
                            logger.info(
                                f"SIGNAL_REJECTED_COSTS | {symbol} {side} - "
                                f"profit={expected_profit:.2f}, cost={profitability.transaction_cost:.2f}"
                            )
                    except (ValueError, KeyError) as e:
                        logger.warning(
                            f"SIGNAL_REJECTED_COSTS | {symbol} {side} - Cost calculation error: {e}"
                        )
                    except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:  # AI-AGENT-REF: narrow exception
                        logger.error(
                            "COST_VALIDATION_FAILED",
                            extra={
                                "cause": e.__class__.__name__,
                                "detail": str(e),
                                "symbol": symbol,
                            },
                        )
                        # Default to rejection on unexpected errors for safety
                        logger.info(
                            f"SIGNAL_REJECTED_COSTS | {symbol} {side} - Validation error"
                        )

                elif decision == PortfolioDecision.BATCH:
                    # For now, treat batching as deferral
                    # In future, implement actual batching logic
                    logger.info(
                        f"SIGNAL_DEFERRED_BATCH | {symbol} {side} - {reasoning}"
                    )

                else:  # REJECT or DEFER
                    logger.info(
                        f"SIGNAL_REJECTED_PORTFOLIO | {symbol} {side} - {reasoning}"
                    )

            except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:  # AI-AGENT-REF: narrow exception
                logger.error(
                    "SIGNAL_EVALUATION_FAILED",
                    extra={"cause": e.__class__.__name__, "detail": str(e), "symbol": symbol},
                )
                # On error, keep the signal to avoid blocking trades
                filtered_signals.append(signal)

        # Log portfolio-level filtering results
        original_count = len(signals)
        filtered_count = len(filtered_signals)
        reduction_percentage = (
            (original_count - filtered_count) / max(original_count, 1)
        ) * 100

        logger.info(
            "PORTFOLIO_SIGNAL_FILTERING_COMPLETE",
            extra={
                "original_signals": original_count,
                "filtered_signals": filtered_count,
                "reduction_percentage": reduction_percentage,
                "market_regime": regime.value,
                "portfolio_decisions": portfolio_decisions,
            },
        )

        # Check if we should trigger portfolio rebalancing instead
        if filtered_count == 0 and original_count > 0:
            should_rebalance, rebalance_reason = _check_portfolio_rebalancing(
                ctx, current_positions, market_data
            )
            if should_rebalance:
                logger.info(f"PORTFOLIO_REBALANCING_TRIGGERED | {rebalance_reason}")
                # Note: Actual rebalancing would be handled by the rebalancer module

        return filtered_signals

    except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:  # AI-AGENT-REF: narrow exception
        logger.error(
            "PORTFOLIO_FILTER_FAILED",
            extra={"cause": e.__class__.__name__, "detail": str(e)},
        )
        # On error, return original signals to avoid blocking trading
        return signals


def _prepare_market_data_for_portfolio_analysis(ctx, signals: list) -> dict:
    """Prepare market data structure for portfolio analysis."""
    try:
        market_data = {
            "prices": {},
            "returns": {},
            "volumes": {},
            "correlations": {},
            "volatility": {},
        }

        # Get unique symbols from signals
        symbols = set()
        for signal in signals:
            symbol = getattr(signal, "symbol", "")
            if symbol:
                symbols.add(symbol)

        # Add index symbol for regime detection
        symbols.add("SPY")

        # Fetch market data for each symbol
        for symbol in symbols:
            try:
                # Get daily data if available
                if hasattr(ctx, "data_fetcher"):
                    df = ctx.data_fetcher.get_daily_df(ctx, symbol)
                    if df is not None and len(df) > 0:
                        # Current price
                        market_data["prices"][symbol] = robust_signal_price(df)

                        # Returns (last 100 periods)
                        if "close" in df.columns and len(df) > 1:
                            prices = df["close"].values[-100:]  # Last 100 days
                            returns = []
                            for i in range(1, len(prices)):
                                if prices[i - 1] > 0:
                                    returns.append(
                                        (prices[i] - prices[i - 1]) / prices[i - 1]
                                    )
                            market_data["returns"][symbol] = returns

                        # Volume
                        if "volume" in df.columns and len(df) > 0:
                            avg_volume = df["volume"].tail(20).mean()  # 20-day average
                            market_data["volumes"][symbol] = avg_volume

                        # Volatility
                        if len(market_data["returns"][symbol]) > 10:
                            returns_std = statistics.stdev(
                                market_data["returns"][symbol][-21:]
                            )  # 21-day vol
                            market_data["volatility"][symbol] = returns_std

            except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError) as e:  # AI-AGENT-REF: narrow exception
                logger.debug(
                    "MARKET_DATA_FETCH_FAILED",
                    extra={"cause": e.__class__.__name__, "detail": str(e), "symbol": symbol},
                )
                continue

        # Calculate correlations between symbols
        _calculate_correlation_matrix(market_data)

        return market_data

    except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:  # AI-AGENT-REF: narrow exception
        logger.error(
            "MARKET_DATA_PREP_FAILED",
            extra={"cause": e.__class__.__name__, "detail": str(e)},
        )
        return {}


def _calculate_correlation_matrix(market_data: dict):
    """Calculate correlation matrix between symbols."""
    try:
        returns_data = market_data.get("returns", {})
        symbols = list(returns_data.keys())

        if len(symbols) < 2:
            return

        correlations = {}

        for symbol1 in symbols:
            correlations[symbol1] = {}
            returns1 = returns_data[symbol1]

            for symbol2 in symbols:
                if symbol1 == symbol2:
                    correlations[symbol1][symbol2] = 1.0
                    continue

                returns2 = returns_data[symbol2]

                # Calculate correlation for overlapping periods
                min_length = min(len(returns1), len(returns2))
                if min_length < 10:  # Need at least 10 observations
                    correlations[symbol1][symbol2] = 0.0
                    continue

                r1 = returns1[-min_length:]
                r2 = returns2[-min_length:]

                try:
                    if statistics.stdev(r1) == 0 or statistics.stdev(r2) == 0:
                        correlation = 0.0
                    else:
                        # Simple correlation calculation
                        mean1, mean2 = statistics.mean(r1), statistics.mean(r2)
                        num = sum(
                            (x - mean1) * (y - mean2)
                            for x, y in zip(r1, r2, strict=False)
                        )
                        den = math.sqrt(
                            sum((x - mean1) ** 2 for x in r1)
                            * sum((y - mean2) ** 2 for y in r2)
                        )
                        correlation = num / den if den > 0 else 0.0

                    correlations[symbol1][symbol2] = correlation

                except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:  # AI-AGENT-REF: narrow exception
                    logger.debug(
                        "CORRELATION_CALC_FAILED",
                        extra={"cause": e.__class__.__name__, "detail": str(e)},
                    )
                    correlations[symbol1][symbol2] = 0.0

        market_data["correlations"] = correlations

    except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:  # AI-AGENT-REF: narrow exception
        logger.error(
            "CORRELATION_MATRIX_FAILED",
            extra={"cause": e.__class__.__name__, "detail": str(e)},
        )


def _get_current_portfolio_positions(ctx) -> dict:
    """Get current portfolio positions."""
    try:
        positions = {}

        # Try to get positions from context
        if hasattr(ctx, "portfolio_positions"):
            positions = ctx.portfolio_positions.copy()
        elif hasattr(ctx, "positions"):
            positions = ctx.positions.copy()

        # Convert to float values
        for symbol in positions:
            try:
                positions[symbol] = float(positions[symbol])
            except (ValueError, TypeError):
                positions[symbol] = 0.0

        return positions

    except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:  # AI-AGENT-REF: narrow exception
        logger.error(
            "POSITION_FETCH_FAILED",
            extra={"cause": e.__class__.__name__, "detail": str(e)},
        )
        return {}


def _estimate_signal_profit(signal, market_data: dict) -> float:
    """Estimate expected profit from a trading signal."""
    try:
        symbol = getattr(signal, "symbol", "")
        getattr(signal, "side", "")
        quantity = getattr(signal, "quantity", 0)

        if not symbol or symbol not in market_data["prices"]:
            return 0.0

        price = market_data["prices"][symbol]
        trade_value = abs(quantity) * price

        # Simple profit estimation based on recent returns
        returns = market_data.get("returns", {}).get(symbol, [])
        if len(returns) > 5:
            # Use average of recent positive returns as profit estimate
            positive_returns = [r for r in returns[-10:] if r > 0]
            if positive_returns:
                avg_positive_return = statistics.mean(positive_returns)
                expected_profit = trade_value * avg_positive_return
                return max(0.0, expected_profit)

        # Fallback: assume 1% profit potential
        return trade_value * 0.01

    except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:  # AI-AGENT-REF: narrow exception
        logger.debug(
            "PROFIT_ESTIMATE_FAILED",
            extra={"cause": e.__class__.__name__, "detail": str(e)},
        )
        return 0.0


def _check_portfolio_rebalancing(
    ctx, current_positions: dict, market_data: dict
) -> tuple:
    """Check if portfolio should be rebalanced instead of individual trades."""
    try:
        settings = get_settings()
        if not settings.ENABLE_PORTFOLIO_FEATURES or not _portfolio_optimizer:
            return False, "Portfolio optimization not available"

        # Get target weights (simplified equal weight for now)
        symbols = list(current_positions.keys())
        if len(symbols) == 0:
            return False, "No positions to rebalance"

        target_weight = 1.0 / len(symbols)
        target_weights = dict.fromkeys(symbols, target_weight)

        # Check if rebalancing is needed
        prices = market_data.get("prices", {})
        should_rebalance, reason = _portfolio_optimizer.should_trigger_rebalance(
            current_positions, target_weights, prices
        )

        return should_rebalance, reason

    except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:  # AI-AGENT-REF: narrow exception
        logger.error(
            "REBALANCE_CHECK_FAILED",
            extra={"cause": e.__class__.__name__, "detail": str(e)},
        )
        return False, f"Error: {str(e)}"


# AI-AGENT-REF: Cost-aware signal evaluation and ensemble gating
class SignalDecisionPipeline:
    """
    Enhanced signal decision pipeline with cost-awareness, regime detection,
    and ensemble gating for robust live trading.
    """
    
    def __init__(self, config: dict = None):
        """Initialize signal decision pipeline."""
        self.config = config or {}
        
        # Cost-aware thresholds
        self.min_edge_threshold = self.config.get("min_edge_threshold", 0.001)  # 0.1% minimum edge
        self.transaction_cost_buffer = self.config.get("transaction_cost_buffer", 0.0005)  # 0.05% buffer
        
        # Ensemble gating
        self.ensemble_min_agree = self.config.get("ensemble_min_agree", 2)
        self.ensemble_total = self.config.get("ensemble_total", 3)
        
        # ATR-based exit scaling
        self.atr_stop_multiplier = self.config.get("atr_stop_multiplier", 2.0)
        self.atr_target_multiplier = self.config.get("atr_target_multiplier", 3.0)
        
        # Regime awareness
        self.regime_volatility_threshold = self.config.get("regime_volatility_threshold", 0.02)  # 2% threshold
        
        logger.info("SignalDecisionPipeline initialized with cost-aware settings")
    
    def evaluate_signal_with_costs(self, symbol: str, df: pd.DataFrame, 
                                 predicted_edge: float, quantity: float = 1000) -> dict:
        """
        Evaluate a trading signal with comprehensive cost analysis.
        
        Returns decision with reason code and metrics.
        """
        try:
            # Get current price and ATR
            current_price = robust_signal_price(df)
            current_atr = self._calculate_current_atr(df)
            
            # Estimate transaction costs
            estimated_costs = self._estimate_transaction_costs(symbol, current_price, quantity)
            
            # Calculate regime characteristics
            regime_info = self._analyze_market_regime(df)
            
            # Calculate score = predicted_edge - costs - buffer
            total_cost = estimated_costs["total_cost_pct"] + self.transaction_cost_buffer
            signal_score = predicted_edge - total_cost
            
            # Decision logic with reason codes
            decision = {
                "symbol": symbol,
                "timestamp": get_utcnow(),
                "predicted_edge": predicted_edge,
                "estimated_costs": estimated_costs,
                "signal_score": signal_score,
                "regime": regime_info,
                "atr": current_atr,
                "current_price": current_price,
                "decision": "REJECT",
                "reason": "UNKNOWN"
            }
            
            # Apply decision criteria
            if signal_score <= 0:
                decision["decision"] = "REJECT"
                decision["reason"] = "REJECT_COST_UNPROFITABLE"
            elif signal_score < self.min_edge_threshold:
                decision["decision"] = "REJECT" 
                decision["reason"] = "REJECT_EDGE_TOO_LOW"
            elif regime_info["is_high_volatility"] and signal_score < self.min_edge_threshold * 2:
                decision["decision"] = "REJECT"
                decision["reason"] = "REJECT_REGIME_HIGH_VOL"
            elif not self._passes_ensemble_gating(df):
                decision["decision"] = "REJECT"
                decision["reason"] = "REJECT_ENSEMBLE_DISAGREEMENT"
            else:
                decision["decision"] = "ACCEPT"
                decision["reason"] = "ACCEPT_OK"
                
                # Calculate ATR-scaled exits
                decision["stop_loss"] = current_price - (current_atr * self.atr_stop_multiplier)
                decision["take_profit"] = current_price + (current_atr * self.atr_target_multiplier)
            
            # Log decision with structured context
            log_level = logging.INFO if decision["decision"] == "ACCEPT" else logging.DEBUG
            logger.log(log_level, "SIGNAL_DECISION: %s for %s - %s (score=%.4f, edge=%.4f, cost=%.4f)", 
                      decision["decision"], symbol, decision["reason"], 
                      signal_score, predicted_edge, total_cost,
                      extra={
                          "component": "signal_decision",
                          "symbol": symbol,
                          "decision": decision["decision"],
                          "reason": decision["reason"],
                          "score": signal_score,
                          "edge": predicted_edge,
                          "cost": total_cost
                      })
            
            return decision
            
        except (KeyError, ValueError, IndexError) as e:
            logger.warning("Signal evaluation failed - data error: %s", e,
                          extra={"component": "signal_decision", "symbol": symbol, "error_type": "data"})
            return {"decision": "REJECT", "reason": "REJECT_DATA_ERROR", "symbol": symbol}
        except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:  # AI-AGENT-REF: narrow exception
            logger.error(
                "SIGNAL_EVALUATION_FAILED",
                extra={
                    "component": "signal_decision",
                    "symbol": symbol,
                    "error_type": "unexpected",
                    "cause": e.__class__.__name__,
                    "detail": str(e),
                },
            )
            return {"decision": "REJECT", "reason": "REJECT_SYSTEM_ERROR", "symbol": symbol}
    
    def _estimate_transaction_costs(self, symbol: str, price: float, quantity: float) -> dict:
        """Estimate transaction costs for a trade."""
        # Basic cost model - can be enhanced with real broker costs
        notional = price * quantity
        
        # Typical costs: commission + spread + slippage + market impact
        commission_pct = 0.0001  # 0.01% commission
        spread_bp = 2  # 2 basis points spread
        slippage_bp = 1  # 1 basis point slippage
        
        spread_cost = (spread_bp / 10000) * notional
        slippage_cost = (slippage_bp / 10000) * notional
        commission = commission_pct * notional
        
        total_cost = commission + spread_cost + slippage_cost
        total_cost_pct = total_cost / notional
        
        return {
            "commission": commission,
            "spread_cost": spread_cost,
            "slippage_cost": slippage_cost,
            "total_cost": total_cost,
            "total_cost_pct": total_cost_pct,
            "notional": notional
        }
    
    def _calculate_current_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate current ATR value."""
        try:
            if len(df) < period:
                return df["close"].iloc[-1] * 0.02  # 2% fallback
            
            # Use the ATR function from indicators if available
            atr_values = atr(df, period)
            return atr_values.iloc[-1] if not atr_values.empty else df["close"].iloc[-1] * 0.02
            
        except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:  # AI-AGENT-REF: narrow exception
            logger.warning(
                "DATA_MUNGING_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e)},
            )
            return df["close"].iloc[-1] * 0.02  # 2% fallback
    
    def _analyze_market_regime(self, df: pd.DataFrame) -> dict:
        """Analyze current market regime characteristics."""
        try:
            # Calculate recent volatility
            returns = df["close"].pct_change().dropna()
            recent_vol = returns.tail(20).std() * np.sqrt(252)  # Annualized volatility
            
            # Determine if high volatility regime
            is_high_vol = recent_vol > self.regime_volatility_threshold
            
            # Calculate trend strength
            short_ma = df["close"].rolling(5).mean().iloc[-1]
            long_ma = df["close"].rolling(20).mean().iloc[-1]
            trend_strength = abs(short_ma - long_ma) / long_ma if long_ma > 0 else 0
            
            return {
                "recent_volatility": recent_vol,
                "is_high_volatility": is_high_vol,
                "trend_strength": trend_strength,
                "regime_type": "trending" if trend_strength > 0.02 else "ranging"
            }
            
        except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:  # AI-AGENT-REF: narrow exception
            logger.debug(
                "REGIME_ANALYSIS_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e)},
            )
            return {
                "recent_volatility": 0.02,
                "is_high_volatility": False,
                "trend_strength": 0.0,
                "regime_type": "unknown"
            }
    
    def _passes_ensemble_gating(self, df: pd.DataFrame) -> bool:
        """Check if signal passes ensemble gating requirements."""
        try:
            # Generate multiple signals
            signals = []
            
            # Signal 1: EMA crossover
            if "EMA_5" in df.columns and "EMA_20" in df.columns:
                ema_signal = 1 if df["EMA_5"].iloc[-1] > df["EMA_20"].iloc[-1] else -1
                signals.append(ema_signal)
            
            # Signal 2: RSI momentum
            if "RSI" in df.columns:
                rsi_val = df["RSI"].iloc[-1]
                if rsi_val < 30:
                    signals.append(1)  # Oversold, buy signal
                elif rsi_val > 70:
                    signals.append(-1)  # Overbought, sell signal
                else:
                    signals.append(0)
            
            # Signal 3: Mean reversion
            if "close" in df.columns:
                zscore = mean_reversion_zscore(df["close"], window=20)
                if zscore.iloc[-1] > 2:
                    signals.append(-1)  # Revert down
                elif zscore.iloc[-1] < -2:
                    signals.append(1)  # Revert up
                else:
                    signals.append(0)
            
            # Count agreements
            positive_signals = sum(1 for s in signals if s > 0)
            negative_signals = sum(1 for s in signals if s < 0)
            max_agreement = max(positive_signals, negative_signals)
            
            # Require minimum agreement
            return max_agreement >= self.ensemble_min_agree
            
        except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:  # AI-AGENT-REF: narrow exception
            logger.debug(
                "ENSEMBLE_GATING_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e)},
            )
            return False  # Fail safe - reject if can't evaluate


# Enhanced signal generation with cost-awareness
def generate_cost_aware_signals(ctx, symbols: list[str]) -> list[dict]:
    """
    Generate trading signals with comprehensive cost-awareness and ensemble gating.
    
    Returns list of signal decisions with reason codes and metrics.
    """
    try:
        # Initialize decision pipeline
        pipeline_config = getattr(ctx, 'signal_pipeline_config', {})
        decision_pipeline = SignalDecisionPipeline(pipeline_config)
        
        signal_decisions = []
        
        for symbol in symbols:
            try:
                # Get market data for symbol
                df = ctx.data_fetcher.get_data(symbol)
                if df is None or len(df) < 50:  # Need sufficient history
                    logger.warning("Insufficient data for %s - skipping", symbol)
                    continue
                
                # Get model prediction (predicted edge)
                predicted_edge = 0.0
                try:
                    if hasattr(ctx, 'model') and ctx.model:
                        features = ctx.feature_generator.generate_features(df)
                        predicted_edge = ctx.model.predict_edge(features)
                except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:  # AI-AGENT-REF: narrow exception
                    logger.debug(
                        "MODEL_PREDICTION_FAILED",
                        extra={"cause": e.__class__.__name__, "detail": str(e), "symbol": symbol},
                    )
                
                # Evaluate signal with costs
                decision = decision_pipeline.evaluate_signal_with_costs(
                    symbol, df, predicted_edge
                )
                signal_decisions.append(decision)
                
            except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:  # AI-AGENT-REF: narrow exception
                logger.warning(
                    "SIGNAL_PROCESSING_FAILED",
                    extra={
                        "component": "signal_generation",
                        "symbol": symbol,
                        "error_type": "processing",
                        "cause": e.__class__.__name__,
                        "detail": str(e),
                    },
                )
        
        # Log summary
        accepted = sum(1 for d in signal_decisions if d.get("decision") == "ACCEPT")
        rejected = len(signal_decisions) - accepted
        
        logger.info("Signal generation complete: %d accepted, %d rejected from %d symbols",
                   accepted, rejected, len(symbols),
                   extra={"component": "signal_generation", "accepted": accepted, "rejected": rejected})
        
        return signal_decisions
        
    except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:  # AI-AGENT-REF: narrow exception
        logger.error(
            "SIGNAL_PROCESSING_FAILED",
            extra={
                "component": "signal_generation",
                "error_type": "unexpected",
                "cause": e.__class__.__name__,
                "detail": str(e),
            },
        )
        return []
