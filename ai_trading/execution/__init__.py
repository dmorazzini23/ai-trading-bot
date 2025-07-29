"""
Execution Module - Institutional Grade Order Management

This module provides comprehensive order execution capabilities for
institutional trading operations including:

- Advanced order lifecycle management and execution algorithms
- Production-ready execution coordination with safety controls
- Liquidity analysis and volume screening
- Slippage monitoring and market impact analysis
- Order routing optimization and execution quality tracking

The module is designed for institutional-scale operations with proper
execution controls, monitoring, and compliance capabilities.
"""

# Import execution components
try:
    from .engine import Order, ExecutionAlgorithm
except ImportError:
    # Create placeholder classes if not available
    class Order:
        pass
    class ExecutionAlgorithm:
        pass

from .production_engine import ProductionExecutionCoordinator, ExecutionResult, OrderRequest
from .liquidity import (
    LiquidityAnalyzer,
    LiquidityManager,
    LiquidityLevel,
    MarketHours
)

# Export all execution classes
__all__ = [
    # Core execution engine
    "Order",
    "ExecutionAlgorithm",
    
    # Production execution coordination
    "ProductionExecutionCoordinator",
    "ExecutionResult",
    "OrderRequest",
    
    # Liquidity management
    "LiquidityAnalyzer",
    "LiquidityManager",
    "LiquidityLevel",
    "MarketHours",
]