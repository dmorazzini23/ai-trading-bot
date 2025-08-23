"""
Risk Management Module - Institutional Grade Risk Controls

This module provides comprehensive risk management capabilities for
institutional trading operations including:

- Kelly Criterion position sizing optimization
- Portfolio risk assessment and monitoring
- Real-time risk controls and alerting
- Value at Risk (VaR) and Expected Shortfall calculations
- Drawdown analysis and recovery monitoring
- Correlation analysis and stress testing
- Circuit breakers and safety mechanisms
- Advanced position sizing algorithms

The module is designed for institutional-scale operations with proper
risk controls, monitoring, and compliance capabilities.
"""
from ai_trading.settings import _DEFAULT_CONFIG as SETTINGS_DEFAULT_CONFIG
from ai_trading.settings import ensure_default_config
from . import kelly as _kelly
from .circuit_breakers import CircuitBreakerState, DeadMansSwitch, DrawdownCircuitBreaker, SafetyLevel, TradingHaltManager, VolatilityCircuitBreaker
from .engine import RiskEngine
from .kelly import InstitutionalKelly, KellyCalculator, KellyCriterion, KellyParams, institutional_kelly
from .manager import PortfolioRiskAssessor, RiskManager
from .metrics import DrawdownAnalyzer, RiskMetricsCalculator
from .position_sizing import ATRPositionSizer, DynamicPositionSizer, PortfolioPositionManager, VolatilityPositionSizer
ensure_default_config()
_kelly._DEFAULT_CONFIG = SETTINGS_DEFAULT_CONFIG
__all__ = ['RiskEngine', 'KellyCriterion', 'KellyCalculator', 'institutional_kelly', 'InstitutionalKelly', 'KellyParams', 'RiskManager', 'PortfolioRiskAssessor', 'ATRPositionSizer', 'VolatilityPositionSizer', 'DynamicPositionSizer', 'PortfolioPositionManager', 'DrawdownCircuitBreaker', 'VolatilityCircuitBreaker', 'TradingHaltManager', 'DeadMansSwitch', 'CircuitBreakerState', 'SafetyLevel', 'RiskMetricsCalculator', 'DrawdownAnalyzer']