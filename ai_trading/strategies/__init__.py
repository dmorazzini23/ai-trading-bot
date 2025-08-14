"""
Canonical strategies public API.
"""
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .base import StrategySignal as TradeSignal
from .moving_average_crossover import (
    MovingAverageCrossoverStrategy,  # AI-AGENT-REF: expose MA crossover strategy
)

# AI-AGENT-REF: expose canonical strategies
__all__ = [
    "MomentumStrategy",
    "MeanReversionStrategy",
    "TradeSignal",
    "MovingAverageCrossoverStrategy",
]
