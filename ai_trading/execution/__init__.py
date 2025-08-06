"""
Execution Module - Institutional Grade Order Management with Enhanced Debugging

This module provides comprehensive order execution capabilities for
institutional trading operations including:

- Advanced order lifecycle management and execution algorithms
- Production-ready execution coordination with safety controls
- Liquidity analysis and volume screening
- Slippage monitoring and market impact analysis
- Order routing optimization and execution quality tracking

ENHANCED FEATURES:
- Complete execution debugging and correlation tracking
- Position reconciliation between bot and broker
- Detailed PnL attribution and explanation system
- Signal-to-execution pipeline visibility

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

# Import enhanced debugging and tracking modules
from .debug_tracker import (
    get_debug_tracker,
    log_signal_to_execution,
    log_execution_phase,
    log_order_outcome,
    log_position_change,
    enable_debug_mode,
    get_execution_statistics,
    ExecutionPhase,
    OrderStatus
)

from .position_reconciler import (
    get_position_reconciler,
    update_bot_position,
    adjust_bot_position,
    force_position_reconciliation,
    start_position_monitoring,
    stop_position_monitoring,
    get_position_discrepancies,
    get_reconciliation_statistics,
    PositionDiscrepancy
)

from .pnl_attributor import (
    get_pnl_attributor,
    update_position_for_pnl,
    record_trade_pnl,
    record_dividend_income,
    get_symbol_pnl_breakdown,
    get_portfolio_pnl_summary,
    explain_recent_pnl_changes,
    get_pnl_attribution_stats,
    PnLSource,
    PnLEvent
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
    
    # Enhanced debugging and tracking
    'get_debug_tracker',
    'log_signal_to_execution',
    'log_execution_phase', 
    'log_order_outcome',
    'log_position_change',
    'enable_debug_mode',
    'get_execution_statistics',
    'ExecutionPhase',
    'OrderStatus',
    
    # Position reconciliation
    'get_position_reconciler',
    'update_bot_position',
    'adjust_bot_position',
    'force_position_reconciliation',
    'start_position_monitoring',
    'stop_position_monitoring',
    'get_position_discrepancies',
    'get_reconciliation_statistics',
    'PositionDiscrepancy',
    
    # PnL attribution
    'get_pnl_attributor',
    'update_position_for_pnl',
    'record_trade_pnl',
    'record_dividend_income',
    'get_symbol_pnl_breakdown',
    'get_portfolio_pnl_summary',
    'explain_recent_pnl_changes',
    'get_pnl_attribution_stats',
    'PnLSource',
    'PnLEvent'
]