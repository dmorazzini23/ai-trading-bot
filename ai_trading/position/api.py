from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
from .market_regime import MarketRegime, detect_market_regime


@dataclass(frozen=True)
class Allocation:
    cash_pct: float
    risk_pct: float = 0.0


class Allocator(Protocol):
    def allocate(self, regime: MarketRegime) -> Allocation: ...
# AI-AGENT-REF: allocation protocol surface
