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
from .debug_tracker import ExecutionPhase, OrderStatus, enable_debug_mode, get_debug_tracker, get_execution_statistics, log_execution_phase, log_order_outcome, log_position_change, log_signal_to_execution
from .engine import ExecutionAlgorithm, ExecutionEngine, Order
from .liquidity import LiquidityAnalyzer, LiquidityLevel, LiquidityManager, MarketHours
from .pnl_attributor import PnLEvent, PnLSource, explain_recent_pnl_changes, get_pnl_attribution_stats, get_pnl_attributor, get_portfolio_pnl_summary, get_symbol_pnl_breakdown, record_dividend_income, record_trade_pnl, update_position_for_pnl
from .position_reconciler import PositionDiscrepancy, adjust_bot_position, force_position_reconciliation, get_position_discrepancies, get_position_reconciler, get_reconciliation_statistics, start_position_monitoring, stop_position_monitoring, update_bot_position
from .transaction_costs import estimate_cost
from .classes import ExecutionResult, OrderRequest
try:
    from .production_engine import ProductionExecutionCoordinator
except (ImportError, AttributeError):
    ProductionExecutionCoordinator = None
__all__ = ['Order', 'ExecutionAlgorithm', 'ExecutionEngine', 'ProductionExecutionCoordinator', 'ExecutionResult', 'OrderRequest', 'LiquidityAnalyzer', 'LiquidityManager', 'LiquidityLevel', 'MarketHours', 'get_debug_tracker', 'log_signal_to_execution', 'log_execution_phase', 'log_order_outcome', 'log_position_change', 'enable_debug_mode', 'get_execution_statistics', 'ExecutionPhase', 'OrderStatus', 'get_position_reconciler', 'update_bot_position', 'adjust_bot_position', 'force_position_reconciliation', 'start_position_monitoring', 'stop_position_monitoring', 'get_position_discrepancies', 'get_reconciliation_statistics', 'PositionDiscrepancy', 'get_pnl_attributor', 'update_position_for_pnl', 'record_trade_pnl', 'record_dividend_income', 'get_symbol_pnl_breakdown', 'get_portfolio_pnl_summary', 'explain_recent_pnl_changes', 'get_pnl_attribution_stats', 'PnLSource', 'PnLEvent', 'estimate_cost']