"""Minimal position package exports."""

from __future__ import annotations

try:
    from .regimes import MarketRegime  # real enum if available
except Exception:  # noqa: BLE001
    from enum import Enum

    class MarketRegime(Enum):
        BULL = "bull"
        BEAR = "bear"
        SIDEWAYS = "sideways"


__all__ = ["MarketRegime"]
