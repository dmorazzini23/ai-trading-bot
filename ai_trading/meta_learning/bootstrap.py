"""Helpers for generating bootstrap training data."""
from __future__ import annotations

from pathlib import Path
from typing import List, Any

from ai_trading.meta_learning import pd


def _generate_bootstrap_training_data(path: str | Path, sample_size: int) -> List[dict[str, Any]]:
    """Return up to ``sample_size`` trade records from *path*.

    The function is intentionally forgiving: errors or missing files yield an
    empty list so callers can continue with fallback strategies.
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if getattr(df, "empty", True):
        return []
    try:
        df = df.dropna()
    except Exception:
        pass
    try:
        records = df.to_dict("records")
    except Exception:
        return []
    return records[:sample_size]


__all__ = ["_generate_bootstrap_training_data"]

