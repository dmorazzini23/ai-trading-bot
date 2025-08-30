"""CLI wrapper for :mod:`ai_trading.algorithm_optimizer`.

This module simply re-exports the package implementation so that legacy
imports continue to function.
"""

from ai_trading.algorithm_optimizer import (  # noqa: F401
    AlgorithmOptimizer,
    MarketConditions,
    MarketRegime,
    OptimizationMetrics,
    OptimizedParameters,
    TradingPhase,
    get_algorithm_optimizer,
    initialize_algorithm_optimizer,
)

__all__ = [
    "AlgorithmOptimizer",
    "MarketConditions",
    "MarketRegime",
    "OptimizationMetrics",
    "OptimizedParameters",
    "TradingPhase",
    "get_algorithm_optimizer",
    "initialize_algorithm_optimizer",
]

if __name__ == "__main__":  # pragma: no cover - simple CLI hint
    import pprint

    pprint.pprint(get_algorithm_optimizer().get_optimization_report())

