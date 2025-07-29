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

# Import institutional-grade components
from . import core
from . import database
from . import risk
from . import strategies
from . import execution
from . import monitoring

# Core institutional exports
from .core import (
    OrderSide, OrderType, OrderStatus, RiskLevel,
    TRADING_CONSTANTS
)

from .risk import (
    KellyCriterion, KellyCalculator, RiskManager
)

from .execution.engine import (
    ExecutionEngine, OrderManager, Order
)

from .strategies.base import (
    BaseStrategy, StrategyRegistry, StrategySignal
)

from .monitoring import (
    MetricsCollector, PerformanceMonitor
)

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