"""Institutional-grade AI trading system."""

from . import rl_trading
from . import core
from . import strategies  
from . import risk

# Institutional-grade exports
from .core import (
    TradePosition, PortfolioMetrics, RiskLimits, TradingSignal,
    MarketData, OrderRequest, ExecutionReport, 
    TradingSide, OrderType, AssetClass, MarketRegime, RiskLevel,
    TradingSystemError, RiskLimitExceededError, DataValidationError, ExecutionError
)
from .strategies import BaseStrategy, TechnicalStrategy, MachineLearningStrategy, EnsembleStrategy
from .risk import KellyCriterion, VolatilityPositionSizing, PositionSizer

__version__ = "1.0.0-institutional"