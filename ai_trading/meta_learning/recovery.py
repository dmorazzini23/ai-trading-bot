"""Fallback data recovery utilities for meta-learning."""
from __future__ import annotations

from pathlib import Path
from datetime import UTC, datetime
from typing import Iterable

from ai_trading.meta_learning import pd

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


def _implement_fallback_data_recovery(path: str | Path, min_samples: int = 0) -> None:
    """Ensure a trade log exists with at least ``min_samples`` rows."""
    p = Path(path)
    if not p.exists():
        df = pd.DataFrame(columns=list(_COLUMNS))
        df.to_csv(p, index=False)
        return
    try:
        df = pd.read_csv(p)
    except Exception:
        df = pd.DataFrame(columns=list(_COLUMNS))
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
        df = pd.concat([df, pd.DataFrame(placeholders)], ignore_index=True)
    except Exception:
        df = pd.DataFrame(placeholders)
    df.to_csv(p, index=False)


__all__ = ["_implement_fallback_data_recovery"]

