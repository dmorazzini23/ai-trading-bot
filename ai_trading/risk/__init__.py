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

# Core risk management components
from .circuit_breakers import (
    CircuitBreakerState,
    DeadMansSwitch,
    DrawdownCircuitBreaker,
    SafetyLevel,
    TradingHaltManager,
    VolatilityCircuitBreaker,
)
from .kelly import KellyCalculator, KellyCriterion
from .manager import PortfolioRiskAssessor, RiskManager
from .position_sizing import (
    ATRPositionSizer,
    DynamicPositionSizer,
    PortfolioPositionManager,
    VolatilityPositionSizer,
)

# Import existing metrics if available
try:
    from .metrics import DrawdownAnalyzer, RiskMetricsCalculator
except ImportError:
    # Create placeholder classes if metrics module doesn't exist
    class RiskMetricsCalculator:
        pass

    class DrawdownAnalyzer:
        pass


# Export all risk management classes
__all__ = [
    # Kelly Criterion position sizing
    "KellyCriterion",
    "KellyCalculator",
    # Risk management and monitoring
    "RiskManager",
    "PortfolioRiskAssessor",
    # Position sizing
    "ATRPositionSizer",
    "VolatilityPositionSizer",
    "DynamicPositionSizer",
    "PortfolioPositionManager",
    # Circuit breakers and safety
    "DrawdownCircuitBreaker",
    "VolatilityCircuitBreaker",
    "TradingHaltManager",
    "DeadMansSwitch",
    "CircuitBreakerState",
    "SafetyLevel",
    # Risk metrics and analysis
    "RiskMetricsCalculator",
    "DrawdownAnalyzer",
]
