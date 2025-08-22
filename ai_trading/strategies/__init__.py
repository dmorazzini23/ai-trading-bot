"""
Canonical strategies public API.
"""
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy

try:  # AI-AGENT-REF: lazy meta-learning import
    from .meta_learning import MetaLearning
except Exception:  # pragma: no cover
    MetaLearning = None  # type: ignore
from .base import StrategySignal as TradeSignal
from .moving_average_crossover import (
    MovingAverageCrossoverStrategy,  # AI-AGENT-REF: expose MA crossover strategy
)

# AI-AGENT-REF: central strategy registry
REGISTRY = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
}
if MetaLearning:
    REGISTRY.update({"meta": MetaLearning, "metalearning": MetaLearning})

# AI-AGENT-REF: expose canonical strategies
__all__ = [
    "MomentumStrategy",
    "MeanReversionStrategy",
    "MetaLearning",
    "TradeSignal",
    "MovingAverageCrossoverStrategy",
    "REGISTRY",
]
