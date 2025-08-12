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

# Import lightweight core components only
from .base import BaseStrategy, StrategySignal

# Provide lazy imports for heavy components to reduce startup time
def get_backtest_engine():
    """Lazy import BacktestEngine to avoid heavy startup costs."""
    from .backtest import BacktestEngine
    return BacktestEngine

def get_multi_timeframe_components():
    """Lazy import multi-timeframe analysis components."""
    from .multi_timeframe import (
        MultiTimeframeAnalyzer,
        MultiTimeframeSignal,
        SignalDirection,
        SignalStrength,
        TimeframeHierarchy,
    )
    return {
        'MultiTimeframeAnalyzer': MultiTimeframeAnalyzer,
        'MultiTimeframeSignal': MultiTimeframeSignal,
        'SignalDirection': SignalDirection,
        'SignalStrength': SignalStrength,
        'TimeframeHierarchy': TimeframeHierarchy,
    }

def get_regime_detection_components():
    """Lazy import regime detection components."""
    from .regime_detection import (
        MarketRegime,
        RegimeDetector,
        TrendStrength,
        VolatilityRegime,
    )
    return {
        'MarketRegime': MarketRegime,
        'RegimeDetector': RegimeDetector,
        'TrendStrength': TrendStrength,
        'VolatilityRegime': VolatilityRegime,
    }

# Export lightweight strategy classes and lazy loading functions
__all__ = [
    # Core lightweight components
    "BaseStrategy",
    "StrategySignal", 
    "TradingSignal",  # For backward compatibility
    # Lazy loading functions
    "get_backtest_engine",
    "get_multi_timeframe_components",
    "get_regime_detection_components",
]
