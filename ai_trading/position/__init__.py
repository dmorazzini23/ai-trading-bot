"""Minimal position package exports."""

from __future__ import annotations

try:
    from .core import MarketRegime  # AI-AGENT-REF: real enum if available
# noqa: BLE001 TODO: narrow exception
except Exception:
    from enum import Enum

    class MarketRegime(Enum):
        BULL = "bull"
        BEAR = "bear"
        SIDEWAYS = "sideways"


__all__ = ["MarketRegime"]
