from __future__ import annotations
from enum import Enum
from collections.abc import Iterable


class MarketRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


def detect_market_regime(
    prices: Iterable[float] | None = None,
) -> MarketRegime:
    """
    Minimal placeholder for tests that import the symbol.
    Does not implement strategy logic; returns UNKNOWN by default.
    """
    return MarketRegime.UNKNOWN
