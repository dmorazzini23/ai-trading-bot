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

# AI-AGENT-REF: Implement lazy imports to prevent import-time crashes
def __getattr__(name):
    """Lazy import submodules to prevent import-time config crashes."""
    if name in ("core", "execution", "strategies", "risk", "monitoring", "database"):
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__version__ = "2.0.0"

__all__ = [
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
    "ExecutionEngine",
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
]