"""
Canonical strategies public API.
"""
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy
from .cross_sectional_momentum import CrossSectionalMomentumStrategy
from .multi_factor_quality_value import MultiFactorQualityValueStrategy
from .pairs_stat_arb import PairsStatArbStrategy
from .pead_event import PEADEventStrategy
from .low_beta_defensive import LowBetaDefensiveStrategy
from .time_series_momentum_overlay import TimeSeriesMomentumOverlayStrategy
try:
    from .meta_learning import MetaLearning
except (KeyError, ValueError, TypeError):
    MetaLearning = None
from .base import StrategySignal as TradeSignal
from .moving_average_crossover import MovingAverageCrossoverStrategy
REGISTRY = {
    'momentum': MomentumStrategy,
    'mean_reversion': MeanReversionStrategy,
    'cross_sectional_momentum': CrossSectionalMomentumStrategy,
    'cross_momentum': CrossSectionalMomentumStrategy,
    'multi_factor_quality_value': MultiFactorQualityValueStrategy,
    'quality_value': MultiFactorQualityValueStrategy,
    'pairs_stat_arb': PairsStatArbStrategy,
    'pairs': PairsStatArbStrategy,
    'pead_event': PEADEventStrategy,
    'pead': PEADEventStrategy,
    'low_beta_defensive': LowBetaDefensiveStrategy,
    'low_beta': LowBetaDefensiveStrategy,
    'time_series_momentum_overlay': TimeSeriesMomentumOverlayStrategy,
    'tsmom_overlay': TimeSeriesMomentumOverlayStrategy,
}
if MetaLearning:
    REGISTRY.update({'meta': MetaLearning, 'metalearning': MetaLearning})
__all__ = [
    'MomentumStrategy',
    'MeanReversionStrategy',
    'CrossSectionalMomentumStrategy',
    'MultiFactorQualityValueStrategy',
    'PairsStatArbStrategy',
    'PEADEventStrategy',
    'LowBetaDefensiveStrategy',
    'TimeSeriesMomentumOverlayStrategy',
    'MetaLearning',
    'TradeSignal',
    'MovingAverageCrossoverStrategy',
    'REGISTRY',
]
