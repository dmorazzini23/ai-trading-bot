"""
Portfolio Optimizer Compatibility Shim

This module provides backward compatibility for code that still imports
from scripts.portfolio_optimizer. The actual implementation has been
moved to ai_trading.portfolio.optimizer.
"""
from ai_trading.portfolio.optimizer import PortfolioDecision, PortfolioMetrics, PortfolioOptimizer, TradeImpactAnalysis, create_portfolio_optimizer
__all__ = ['PortfolioDecision', 'PortfolioOptimizer', 'PortfolioMetrics', 'TradeImpactAnalysis', 'create_portfolio_optimizer']