"""
Canonical strategies public API.
"""
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy
try:
    from .meta_learning import MetaLearning
except (KeyError, ValueError, TypeError):
    MetaLearning = None
from .base import StrategySignal as TradeSignal
from .moving_average_crossover import MovingAverageCrossoverStrategy
REGISTRY = {'momentum': MomentumStrategy, 'mean_reversion': MeanReversionStrategy}
if MetaLearning:
    REGISTRY.update({'meta': MetaLearning, 'metalearning': MetaLearning})
__all__ = ['MomentumStrategy', 'MeanReversionStrategy', 'MetaLearning', 'TradeSignal', 'MovingAverageCrossoverStrategy', 'REGISTRY']