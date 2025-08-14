"""
Canonical strategies public API.
"""
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .base import StrategySignal as TradeSignal

# AI-AGENT-REF: expose canonical strategies
__all__ = ["MomentumStrategy", "MeanReversionStrategy", "TradeSignal"]
