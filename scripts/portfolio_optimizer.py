# AI-AGENT-REF: Compatibility shim - module moved to ai_trading.portfolio.optimizer
"""
Portfolio Optimizer Compatibility Shim

This module provides backward compatibility for code that still imports
from scripts.portfolio_optimizer. The actual implementation has been
moved to ai_trading.portfolio.optimizer.
"""

from ai_trading.portfolio.optimizer import (
    PortfolioDecision,
    PortfolioOptimizer,
    PortfolioMetrics,
    TradeImpactAnalysis,
    create_portfolio_optimizer
)

# Re-export all classes for backward compatibility
__all__ = [
    "PortfolioDecision",
    "PortfolioOptimizer", 
    "PortfolioMetrics",
    "TradeImpactAnalysis",
    "create_portfolio_optimizer"
]
