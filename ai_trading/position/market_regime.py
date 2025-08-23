from __future__ import annotations
from enum import Enum, auto


class MarketRegime(Enum):
    UNKNOWN = auto()
    BULL = auto()
    BEAR = auto()
    SIDEWAYS = auto()


def detect_market_regime(*_args, **_kwargs) -> MarketRegime:
    """Placeholder to keep imports stable during tests."""
    return MarketRegime.UNKNOWN
# AI-AGENT-REF: stable market regime enum for test imports
