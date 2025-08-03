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
    weight: float = 0.0  # AI-AGENT-REF: Default to 0.0, will be set by portfolio management  
    asset_class: str = "equity"  # AI-AGENT-REF: Moved after weight to fix parameter order
    price: float = 0.0


class Strategy:
    """Base strategy interface."""

    name: str = "base"
    asset_class: str = "equity"

    def generate(self, ctx) -> List[TradeSignal]:
        """Return a list of TradeSignal objects."""
        return []
