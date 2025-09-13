"""Fallback data recovery utilities for meta-learning."""
from __future__ import annotations

from pathlib import Path
from datetime import UTC, datetime
from typing import Iterable
from types import ModuleType

from ai_trading.utils.lazy_imports import load_pandas

pd: ModuleType | None = load_pandas()

_COLUMNS: Iterable[str] = [
    "timestamp",
    "symbol",
    "side",
    "entry_price",
    "exit_price",
    "quantity",
    "pnl",
    "signal_tags",
]


def _require_pandas() -> ModuleType:
    """Return :mod:`pandas` or raise if unavailable."""
    if pd is None:
        raise RuntimeError("pandas is required for meta-learning recovery")
    return pd


def recover_dataframe(path: str | Path) -> "pd.DataFrame":
    """Load a DataFrame from ``path`` ensuring pandas is available."""
    pandas = _require_pandas()
    return pandas.read_csv(path)


def _implement_fallback_data_recovery(path: str | Path, min_samples: int = 0) -> None:
    """Ensure a trade log exists with at least ``min_samples`` rows."""
    pandas = _require_pandas()
    p = Path(path)
    if not p.exists():
        df = pandas.DataFrame(columns=list(_COLUMNS))
        df.to_csv(p, index=False)
        return
    try:
        df = pandas.read_csv(p)
    except Exception:
        df = pandas.DataFrame(columns=list(_COLUMNS))
    if len(df) >= min_samples:
        return
    # Append placeholder rows to reach the threshold
    rows_needed = max(0, min_samples - len(df))
    now = datetime.now(UTC).isoformat()
    placeholders = [
        {
            "timestamp": now,
            "symbol": "DUMMY",
            "side": "buy",
            "entry_price": 0.0,
            "exit_price": 0.0,
            "quantity": 0,
            "pnl": 0.0,
            "signal_tags": "fallback",
        }
        for _ in range(rows_needed)
    ]
    try:
        df = pandas.concat([df, pandas.DataFrame(placeholders)], ignore_index=True)
    except Exception:
        df = pandas.DataFrame(placeholders)
    df.to_csv(p, index=False)


__all__ = ["recover_dataframe", "_implement_fallback_data_recovery"]

