"""Indicator helpers for signal generation."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ai_trading.logging import get_logger
from ai_trading.utils.lazy_imports import load_pandas, load_pandas_ta

logger = get_logger(__name__)


def psar(df: Any) -> Any:
    """Compute Parabolic SAR indicator columns.

    Parameters
    ----------
    df : Any
        DataFrame with ``high``, ``low`` and ``close`` columns.

    Returns
    -------
    Any
        Copy of ``df`` with ``psar_long`` and ``psar_short`` columns.
    """
    pd = load_pandas()
    required = {"high", "low", "close"}
    cols = set(getattr(df, "columns", []))
    missing = required - cols
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"DataFrame missing required column(s): {missing_cols}")
    if len(df) == 0:
        raise ValueError("DataFrame is empty")

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
