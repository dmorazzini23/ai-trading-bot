from __future__ import annotations
from typing import Optional

DEFAULT_CAPITAL_CAP = 0.04
DEFAULT_MAX_POSITION_FALLBACK = 8000.0


def derive_cap_from_settings(
    equity: Optional[float],
    capital_cap: float = DEFAULT_CAPITAL_CAP,
    fallback: float = DEFAULT_MAX_POSITION_FALLBACK,
) -> float:
    if equity and equity > 0:
        return float(equity) * float(capital_cap)
    return float(fallback)


def clamp_position_size(
    size: float,
    *,
    min_size: float = 0.0,
    max_size: Optional[float] = None,
) -> float:
    if max_size is not None:
        size = min(size, max_size)
    return max(size, min_size)


__all__ = [
    "derive_cap_from_settings",
    "clamp_position_size",
    "DEFAULT_CAPITAL_CAP",
    "DEFAULT_MAX_POSITION_FALLBACK",
]

