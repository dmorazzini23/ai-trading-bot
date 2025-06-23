"""Helpers for logging model performance metrics."""

from __future__ import annotations

import csv
import logging
import os
from typing import Any, Sequence

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
