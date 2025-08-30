"""Execution Module - Institutional Grade Order Management with Enhanced Debugging."""

from __future__ import annotations

# Core exports that should always be available
from .classes import ExecutionResult, OrderRequest
from .engine import ExecutionAlgorithm, ExecutionEngine, Order

# Optional submodule: algorithms
try:  # pragma: no cover - optional dependency
    from . import algorithms
except Exception:  # noqa: BLE001 - broad to guard optional deps
    algorithms = None

# Optional utilities guarded against missing dependencies
try:  # pragma: no cover - optional dependency
    from .debug_tracker import (
        ExecutionPhase,
        OrderStatus,
        enable_debug_mode,
        get_debug_tracker,
        get_execution_statistics,
        log_execution_phase,
        log_order_outcome,
        log_position_change,
        log_signal_to_execution,
    )
except Exception:  # noqa: BLE001 - broad to guard optional deps
    ExecutionPhase = OrderStatus = None
    enable_debug_mode = get_debug_tracker = None
    get_execution_statistics = None
    log_execution_phase = log_order_outcome = None
    log_position_change = log_signal_to_execution = None

try:  # pragma: no cover - optional dependency
    from .liquidity import (
        LiquidityAnalyzer,
        LiquidityLevel,
        LiquidityManager,
        MarketHours,
    )
except Exception:  # noqa: BLE001 - broad to guard optional deps
    LiquidityAnalyzer = LiquidityLevel = None
    LiquidityManager = MarketHours = None

try:  # pragma: no cover - optional dependency
    from .pnl_attributor import (
        PnLEvent,
        PnLSource,
        explain_recent_pnl_changes,
        get_pnl_attribution_stats,
        get_pnl_attributor,
        get_portfolio_pnl_summary,
        get_symbol_pnl_breakdown,
        record_dividend_income,
        record_trade_pnl,
        update_position_for_pnl,
    )
except Exception:  # noqa: BLE001 - broad to guard optional deps
    PnLEvent = PnLSource = None
    explain_recent_pnl_changes = get_pnl_attribution_stats = None
    get_pnl_attributor = get_portfolio_pnl_summary = None
    get_symbol_pnl_breakdown = record_dividend_income = None
    record_trade_pnl = update_position_for_pnl = None

try:  # pragma: no cover - optional dependency
    from .position_reconciler import (
        PositionDiscrepancy,
        adjust_bot_position,
        force_position_reconciliation,
        get_position_discrepancies,
        get_position_reconciler,
        get_reconciliation_statistics,
        start_position_monitoring,
        stop_position_monitoring,
        update_bot_position,
    )
except Exception:  # noqa: BLE001 - broad to guard optional deps
    PositionDiscrepancy = None
    adjust_bot_position = force_position_reconciliation = None
    get_position_discrepancies = get_position_reconciler = None
    get_reconciliation_statistics = start_position_monitoring = None
    stop_position_monitoring = update_bot_position = None

try:  # pragma: no cover - optional dependency
    from .transaction_costs import estimate_cost
except Exception:  # noqa: BLE001 - broad to guard optional deps
    estimate_cost = None

try:  # pragma: no cover - optional dependency
    from .production_engine import ProductionExecutionCoordinator
except Exception:  # noqa: BLE001 - broad to guard optional deps
    ProductionExecutionCoordinator = None

__all__ = [
    "Order",
    "ExecutionAlgorithm",
    "ExecutionEngine",
    "ProductionExecutionCoordinator",
    "ExecutionResult",
    "OrderRequest",
    "algorithms",
    "LiquidityAnalyzer",
    "LiquidityManager",
    "LiquidityLevel",
    "MarketHours",
    "get_debug_tracker",
    "log_signal_to_execution",
    "log_execution_phase",
    "log_order_outcome",
    "log_position_change",
    "enable_debug_mode",
    "get_execution_statistics",
    "ExecutionPhase",
    "OrderStatus",
    "get_position_reconciler",
    "update_bot_position",
    "adjust_bot_position",
    "force_position_reconciliation",
    "start_position_monitoring",
    "stop_position_monitoring",
    "get_position_discrepancies",
    "get_reconciliation_statistics",
    "PositionDiscrepancy",
    "get_pnl_attributor",
    "update_position_for_pnl",
    "record_trade_pnl",
    "record_dividend_income",
    "get_symbol_pnl_breakdown",
    "get_portfolio_pnl_summary",
    "explain_recent_pnl_changes",
    "get_pnl_attribution_stats",
    "PnLSource",
    "PnLEvent",
    "estimate_cost",
]

