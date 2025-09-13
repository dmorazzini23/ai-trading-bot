"""Indicator helpers for signal generation."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ai_trading.logging import get_logger
from ai_trading.utils.lazy_imports import load_pandas, load_pandas_ta

logger = get_logger(__name__)

REQUIRED_COLS = {"open", "high", "low", "close", "volume"}


def _validate_ohlcv(df: Any):
    """Validate OHLCV DataFrame structure and types.

    Raises ``KeyError`` if required columns are missing and ``ValueError`` if
    the ``close`` column contains non-numeric data.
    """

    pd = load_pandas()
    cols = set(getattr(df, "columns", []))
    missing = REQUIRED_COLS - cols
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise KeyError(f"DataFrame missing required column(s): {missing_cols}")
    if getattr(df, "empty", False):
        logger.warning("EMPTY_DATAFRAME_FOR_INDICATOR")
        return df
    try:
        df = df.copy()
        df["close"] = pd.to_numeric(df["close"], errors="raise")
    except Exception as exc:
        raise ValueError("Column 'close' must be numeric") from exc
    return df


def psar(df: Any) -> Any:
    """Compute Parabolic SAR indicator columns.

    Parameters
    ----------
    df : Any
        DataFrame with ``open``, ``high``, ``low``, ``close`` and ``volume`` columns.

    Returns
    -------
    Any
        Copy of ``df`` with ``psar_long`` and ``psar_short`` columns.
    """

    pd = load_pandas()
    df = _validate_ohlcv(df)
    if df.empty:
        return df

    ta = load_pandas_ta()
    if pd is None or ta is None:
        logger.warning("PANDAS_TA_PSAR_MISSING")
        df = df.copy()
        approx = ((df["high"] + df["low"]) / 2).astype(float)
        df["psar_long"] = approx
        df["psar_short"] = approx
        return df
    df = df.copy()
    try:
        psar_df = ta.psar(df["high"], df["low"], df["close"])
        df["psar_long"] = psar_df["PSARl_0.02_0.2"].astype(float)
        df["psar_short"] = psar_df["PSARs_0.02_0.2"].astype(float)
    except Exception as exc:  # pragma: no cover - optional dependency behaviour
        logger.warning("PSAR_CALC_FAILED", extra={"cause": exc.__class__.__name__})
        approx = ((df["high"] + df["low"]) / 2).astype(float)
        df["psar_long"] = approx
        df["psar_short"] = approx
    return df


def composite_signal_confidence(confidences: Mapping[str, float] | Iterable[float]) -> float:
    """Combine signal confidences into a composite score.

    Parameters
    ----------
    confidences : Mapping[str, float] | Iterable[float]
        Mapping of signal labels to confidence values. If an iterable is
        provided, it is converted into a mapping using enumeration.

    Returns
    -------
    float
        Sum of confidence values.
    """
    if not isinstance(confidences, Mapping):
        confidences = {str(i): c for i, c in enumerate(confidences)}
    return float(sum(float(c) for c in confidences.values()))


__all__ = ["psar", "composite_signal_confidence"]
