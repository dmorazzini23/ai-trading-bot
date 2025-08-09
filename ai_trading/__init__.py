"""
AI Trading Bot Module - Institutional Grade Trading Platform

This module contains the core trading bot functionality and institutional-grade
components for professional trading operations including:

- Core trading enums and constants
- Database models and connection management  
- Kelly Criterion risk management
- Strategy framework and execution engine
- Performance monitoring and alerting
- Institutional-grade order management

The platform is designed for institutional-scale operations with proper
risk controls, monitoring, and compliance capabilities.
"""

# Public API re-exports - canonical import path
from .execution.engine import ExecutionEngine

# AI-AGENT-REF: Implement lazy imports to prevent import-time crashes
def __getattr__(name):
    """Lazy import submodules to prevent import-time config crashes."""
    if name in ("core", "execution", "strategies", "risk", "monitoring", "database", "workers"):
        import importlib
        return importlib.import_module(f".{name}", __name__)
    elif name in _moved_modules:
        # Lazy import for moved modules
        return _lazy_import_module(name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__version__ = "2.0.0"

# Public, convenience re-exports (optional, for nicer DX)
# Use lazy imports to prevent import-time crashes
def _lazy_import_module(name):
    """Lazy import helper for moved modules."""
    import importlib
    return importlib.import_module(f".{name}", __name__)

# Setup lazy attributes for moved modules
_moved_modules = ["signals", "data_fetcher", "indicators", "pipeline", "portfolio", "rebalancer"]

__all__ = [
    "ExecutionEngine",
] + sorted(set(list(globals().get("__all__", [])) + [
    # Core trading infrastructure
    "OrderSide",
    "OrderType", 
    "OrderStatus",
    "RiskLevel",
    "TRADING_CONSTANTS",
    
    # Risk management
    "KellyCriterion",
    "KellyCalculator", 
    "RiskManager",
    
    # Execution engine
    "OrderManager",
    "Order",
    
    # Strategy framework
    "BaseStrategy",
    "StrategyRegistry",
    "StrategySignal",
    
    # Monitoring and metrics
    "MetricsCollector",
    "PerformanceMonitor",
    
    # Submodules
    "core",
    "database",
    "risk", 
    "strategies",
    "execution",
    "monitoring",
    
    # Moved modules
    "signals", "data_fetcher", "indicators", "pipeline", "portfolio", "rebalancer",
]))