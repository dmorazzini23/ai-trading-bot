"""Minimal position package exports."""

from __future__ import annotations

try:  # pragma: no cover - best effort
    from .regime import MarketRegime
except Exception:  # pragma: no cover - fallback
    from enum import Enum

    class MarketRegime(Enum):
        BULL = "bull"
        BEAR = "bear"
        NEUTRAL = "neutral"


__all__ = ["MarketRegime"]
