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
from .momentum import MomentumStrategy
from importlib import import_module

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
    # Common strategies and helpers
    "MomentumStrategy",
    "asset_class_for",
    # Public strategy exports
    "MeanReversionStrategy",
]


def _resolve(name: str):
    try:
        return globals()[name]
    except KeyError:
        pass
    candidates = [
        ("ai_trading.strategies.mean_reversion", "MeanReversionStrategy"),
        ("ai_trading.strategies.meanreversion", "MeanReversionStrategy"),
        ("ai_trading.strategies.mean_reversion", "MeanReversion"),
        ("ai_trading.strategies.mean_reversion", "MeanRevesionStrategy"),
        ("strategies.mean_reversion", "MeanReversionStrategy"),
    ]
    for modpath, attr in candidates:
        try:
            m = import_module(modpath)
            if hasattr(m, attr):
                return getattr(m, attr)
        except Exception:
            continue
    raise ImportError(
        "Could not resolve 'MeanReversionStrategy' from ai_trading.strategies. "
        "Ensure the implementation class exists in strategies/mean_reversion.py "
        "and is named 'MeanReversionStrategy'."
    )


MeanReversionStrategy = _resolve("MeanReversionStrategy")


def asset_class_for(symbol: str) -> str:
    """Map tickers to a basic asset class."""
    sym = symbol.upper()
    if sym.endswith("USD") and len(sym) == 6:
        return "forex"
    if sym.startswith(("BTC", "ETH")):
        return "crypto"
    return "equity"
