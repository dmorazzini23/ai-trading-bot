"""
Portfolio construction and position sizing modules.
Exports:
    compute_portfolio_weights(ctx, symbols) -> Dict[str, float]
    PortfolioDecision, PortfolioOptimizer, create_portfolio_optimizer
    optimize_equal_weight
"""

from .weights import compute_portfolio_weights  # re-export for stable import path
from .optimizer import (
    PortfolioDecision,
    PortfolioOptimizer,
    create_portfolio_optimizer,
    optimize_equal_weight,
)

__all__ = [
    "compute_portfolio_weights",
    "PortfolioDecision",
    "PortfolioOptimizer",
    "create_portfolio_optimizer",
    "optimize_equal_weight",
]
