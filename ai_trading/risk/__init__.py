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

The module is designed for institutional-scale operations with proper
risk controls, monitoring, and compliance capabilities.
"""

# Core risk management components
from .kelly import KellyCriterion, KellyCalculator
from .manager import RiskManager
from .metrics import RiskMetricsCalculator, DrawdownAnalyzer

# Export all risk management classes
__all__ = [
    # Kelly Criterion position sizing
    "KellyCriterion",
    "KellyCalculator",
    
    # Risk management and monitoring
    "RiskManager",
    
    # Risk metrics and analysis
    "RiskMetricsCalculator",
    "DrawdownAnalyzer",
]