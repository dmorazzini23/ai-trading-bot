from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

def compute_max_drawdown(curve: Sequence[float]) -> float:
    """Return max peak-to-trough drawdown as a fraction."""
    if not curve:
        return 0.0
    peak = float(curve[0])
    mdd = 0.0
    for x in curve:
        try:
            v = float(x)
        except (TypeError, ValueError):
            continue
        if v > peak:
            peak = v
        if peak:
            mdd = max(mdd, (peak - v) / peak)
    return mdd

def _ensure_parent(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning("Could not create parent dir for %s: %s", path, e)

def _write_csv_row(filename: str, row: Mapping[str, Any]) -> None:
    """Write a single row to CSV file with headers if file doesn't exist."""
    import os
    path = Path(filename)
    _ensure_parent(path)
    
    try:
        file_exists = path.exists() and path.stat().st_size > 0
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: "" if v is None else str(v) for k, v in row.items()})
    except (OSError, csv.Error) as exc:
        logger.warning("Failed to update metrics file %s: %s", filename, exc)

def log_metrics(
    record: Mapping[str, Any],
    filename: str = "metrics/model_performance.csv",
    equity_curve: Sequence[float] | None = None,
) -> None:
    """Append a metrics record to ``filename``.
    
    Parameters
    ----------
    record : Mapping[str, Any]
        Dictionary of metric values to log.
    filename : str, optional
        Output CSV path, by default ``"metrics/model_performance.csv"``.
    equity_curve : Sequence[float] | None, optional
        Optional list of portfolio values used to compute the ``max_drawdown``
        metric. When provided the computed value is added to ``record`` if not
        already present.
    """
    row = dict(record)
    if equity_curve and "max_drawdown" not in row:
        try:
            row["max_drawdown"] = compute_max_drawdown(equity_curve)
        except Exception:
            row["max_drawdown"] = ""
    _write_csv_row(filename, row)

def log_volatility(value: float, filename: str | None = None) -> None:
    if filename:
        _write_csv_row(filename, {"volatility": value})
    else:
        logger.info("volatility=%s", value)

def log_regime_toggle(tag: str, regime: str, filename: str | None = None) -> None:
    row = {"tag": tag, "regime": regime}
    if filename:
        _write_csv_row(filename, row)
    else:
        logger.info("regime_toggle tag=%s regime=%s", tag, regime)