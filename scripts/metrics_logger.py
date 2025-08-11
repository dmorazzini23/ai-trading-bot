"""Helpers for logging model performance metrics."""

from __future__ import annotations

import csv
import logging
import os
from collections.abc import Sequence
from typing import Any

# AI-AGENT-REF: guard numpy import for test environments
import numpy as np

logger = logging.getLogger(__name__)


def compute_max_drawdown(equity_curve: Sequence[float]) -> float:
    """Return the maximum drawdown for ``equity_curve``.

    Parameters
    ----------
    equity_curve : Sequence[float]
        Sequence of portfolio values or equity amounts.

    Returns
    -------
    float
        Maximum drawdown as a fraction between 0 and 1.
    """

    if not equity_curve:
        return 0.0

    arr = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(arr)
    drawdowns = (peak - arr) / peak
    return float(np.max(drawdowns))


def log_metrics(
    record: dict[str, Any],
    filename: str = "metrics/model_performance.csv",
    equity_curve: Sequence[float] | None = None,
) -> None:
    """Append a metrics record to ``filename``.

    Parameters
    ----------
    record : dict[str, Any]
        Dictionary of metric values to log.
    filename : str, optional
        Output CSV path, by default ``"metrics/model_performance.csv"``.
    equity_curve : Sequence[float] | None, optional
        Optional list of portfolio values used to compute the ``max_drawdown``
        metric. When provided the computed value is added to ``record`` if not
        already present.
    """

    if equity_curve and "max_drawdown" not in record:
        record["max_drawdown"] = compute_max_drawdown(equity_curve)

    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        file_exists = os.path.isfile(filename)
        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)
    except (OSError, csv.Error) as exc:  # pragma: no cover - best effort logging
        logger.warning("Failed to update metrics file %s: %s", filename, exc)


def log_volatility(value: float) -> None:
    """Log volatility reading."""
    logger.info("VOLATILITY_READING", extra={"value": float(value)})


def log_atr_stop(symbol: str, stop: float) -> None:
    """Log ATR stop level."""
    logger.info("ATR_STOP", extra={"symbol": symbol, "stop": float(stop)})


def log_pyramid_add(symbol: str, position: float) -> None:
    """Log a pyramiding position add."""
    logger.info("PYRAMID_ADD", extra={"symbol": symbol, "position": position})


def log_regime_toggle(symbol: str, regime: str) -> None:
    """Log regime changes."""
    logger.info("REGIME_TOGGLE", extra={"symbol": symbol, "regime": regime})
