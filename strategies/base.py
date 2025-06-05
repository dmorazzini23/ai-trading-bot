from dataclasses import dataclass
from typing import List


def asset_class_for(symbol: str) -> str:
    sym = symbol.upper()
    if sym.endswith("USD") and len(sym) == 6:
        return "forex"
    if sym.startswith("BTC") or sym.startswith("ETH"):
        return "crypto"
    return "equity"

@dataclass
class TradeSignal:
    symbol: str
    side: str  # 'buy' or 'sell'
    confidence: float
    strategy: str
    asset_class: str = "equity"
    weight: float = 1.0
    price: float = 0.0

class Strategy:
    """Base strategy interface."""

    name: str = "base"
    asset_class: str = "equity"

    def generate(self, ctx) -> List[TradeSignal]:
        """Return a list of TradeSignal objects."""
        return []
