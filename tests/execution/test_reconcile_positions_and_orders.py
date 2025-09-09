from datetime import UTC, datetime

from ai_trading.execution.reconcile import ReconciliationResult, reconcile_positions_and_orders


def test_reconcile_positions_and_orders_has_timestamp():
    """reconcile_positions_and_orders should populate reconciled_at timestamp."""
    result = reconcile_positions_and_orders()
    assert isinstance(result, ReconciliationResult)
    assert isinstance(result.reconciled_at, datetime)
    assert result.reconciled_at.tzinfo is UTC
    assert result.position_drifts == []
    assert result.order_drifts == []
    assert result.actions_taken == []

