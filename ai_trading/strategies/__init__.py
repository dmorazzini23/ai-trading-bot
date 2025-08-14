"""
Canonical strategies public API.
"""
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .meta_learning import MetaLearning
from .base import StrategySignal as TradeSignal
from .moving_average_crossover import (
    MovingAverageCrossoverStrategy,  # AI-AGENT-REF: expose MA crossover strategy
)

# AI-AGENT-REF: central strategy registry
REGISTRY = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "meta": MetaLearning,
    "metalearning": MetaLearning,
}

# AI-AGENT-REF: expose canonical strategies
__all__ = [
    "MomentumStrategy",
    "MeanReversionStrategy",
    "MetaLearning",
    "TradeSignal",
    "MovingAverageCrossoverStrategy",
    "REGISTRY",
]
