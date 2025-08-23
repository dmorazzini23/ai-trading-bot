"""Minimal position package exports."""
from __future__ import annotations
try:
    from .core import MarketRegime
except (KeyError, ValueError, TypeError):
    from enum import Enum

    class MarketRegime(Enum):
        BULL = 'bull'
        BEAR = 'bear'
        SIDEWAYS = 'sideways'
__all__ = ['MarketRegime']