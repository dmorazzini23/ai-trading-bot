"""
Advanced Trading Strategies Module - Institutional Grade Strategy Framework

This module provides comprehensive trading strategy capabilities for
institutional trading operations including:

- Multi-timeframe analysis and signal generation
- Market regime detection and adaptation
- Advanced strategy orchestration and coordination
- Signal combination and conflict resolution
- Adaptive algorithm frameworks

The module is designed for institutional-scale operations with proper
strategy diversification, risk management, and performance monitoring.
"""

# Import strategy components
from .multi_timeframe import (
    MultiTimeframeAnalyzer,
    MultiTimeframeSignal,
    SignalDirection,
    SignalStrength,
    TimeframeHierarchy,
)
from .regime_detection import (
    MarketRegime,
    RegimeDetector,
    TrendStrength,
    VolatilityRegime,
)

# Import existing strategy components if available
try:
    from .base import BaseStrategy, StrategySignal
except ImportError:
    # Create placeholder classes if not available
    class BaseStrategy:
        pass

    class StrategySignal:
        pass


try:
    from .ai_trading.signals import TradingSignal
except ImportError:

    class TradingSignal:
        pass


try:
    from .backtest import BacktestEngine
except ImportError:

    class BacktestEngine:
        pass


# Export all strategy classes
__all__ = [
    # New advanced strategy components
    "MultiTimeframeAnalyzer",
    "MultiTimeframeSignal",
    "TimeframeHierarchy",
    "SignalStrength",
    "SignalDirection",
    "RegimeDetector",
    "MarketRegime",
    "VolatilityRegime",
    "TrendStrength",
    # Legacy strategy components
    "BaseStrategy",
    "StrategySignal",
    "TradingSignal",
    "BacktestEngine",
]
