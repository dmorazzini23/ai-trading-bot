"""
Advanced position management package for intelligent position holding strategies.

This package implements sophisticated position management with:
- Dynamic trailing stops based on volatility and momentum
- Multi-tiered profit taking with scale-out strategies  
- Market regime-aware position management
- Technical signal integration for exit timing
- Portfolio-level correlation and exposure management

AI-AGENT-REF: Advanced intelligent position management system
"""

from .intelligent_manager import IntelligentPositionManager
from .market_regime import MarketRegimeDetector
from .technical_analyzer import TechnicalSignalAnalyzer
from .trailing_stops import TrailingStopManager
from .profit_taking import ProfitTakingEngine
from .correlation_analyzer import PortfolioCorrelationAnalyzer

__all__ = [
    'IntelligentPositionManager',
    'MarketRegimeDetector', 
    'TechnicalSignalAnalyzer',
    'TrailingStopManager',
    'ProfitTakingEngine',
    'PortfolioCorrelationAnalyzer'
]