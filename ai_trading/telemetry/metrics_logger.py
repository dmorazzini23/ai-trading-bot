from __future__ import annotations

import csv
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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
        peak = max(peak, v)
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
    """Append a metrics record to ``filename``."""
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


def log_atr_stop(symbol: str, stop: float) -> None:
    """No-op placeholder for ATR stop telemetry."""
    return None


def log_pyramid_add(symbol: str, new_pos: float) -> None:
    """No-op placeholder for pyramid add telemetry."""
    return None
