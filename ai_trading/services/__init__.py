"""Operator-facing service facades over the canonical runtime modules."""

from ai_trading.services.control_plane import ControlPlaneService
from ai_trading.services.execution import (
    ExecutionService,
    execute_signal_orders,
    execute_trade_cycle,
    submit_order,
)
from ai_trading.services.governance import GovernanceService
from ai_trading.services.portfolio import (
    PortfolioService,
    compute_portfolio_weights,
    ensure_portfolio_weights,
)
from ai_trading.services.reconciliation import (
    ReconciliationService,
    require_success,
    reconcile_position_targets,
)
from ai_trading.services.risk_approval import RiskApprovalService
from ai_trading.services.signal import (
    evaluate_signal_and_confirm,
    generate_directional_signals,
)

__all__ = [
    "ControlPlaneService",
    "GovernanceService",
    "RiskApprovalService",
    "ExecutionService",
    "PortfolioService",
    "ReconciliationService",
    "compute_portfolio_weights",
    "ensure_portfolio_weights",
    "evaluate_signal_and_confirm",
    "execute_signal_orders",
    "execute_trade_cycle",
    "generate_directional_signals",
    "require_success",
    "reconcile_position_targets",
    "submit_order",
]
