from datetime import UTC, datetime
from types import SimpleNamespace

from ai_trading.execution.reconcile import ReconciliationResult, reconcile_positions_and_orders
from ai_trading.core.interfaces import Order, OrderStatus, OrderType
from ai_trading.order.types import OrderSide


def test_reconcile_positions_and_orders_has_timestamp():
    """reconcile_positions_and_orders should populate reconciled_at timestamp."""
    result = reconcile_positions_and_orders()
    assert isinstance(result, ReconciliationResult)
    assert isinstance(result.reconciled_at, datetime)
    assert result.reconciled_at.tzinfo is UTC
    assert result.position_drifts == []
    assert result.order_drifts == []
    assert result.actions_taken == []


class DummyBroker:
    """Minimal broker API used for reconciliation tests."""

    def __init__(self):
        self._order = SimpleNamespace(
            id="1", status="filled", filled_qty=10, symbol="AAPL"
        )

    def list_positions(self):
        return [
            SimpleNamespace(
                symbol="AAPL", qty=10, market_value=0, cost_basis=0, unrealized_pl=0
            )
        ]

    def list_orders(self, status="open"):
        return []

    def get_order(self, order_id):
        return self._order


def test_reconciliation_updates_state():
    """Reconciliation should update holdings and local order statuses."""
    broker = DummyBroker()
    ctx = SimpleNamespace(
        api=broker,
        positions={"AAPL": 0},
        orders=[
            Order(
                id="1",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                status=OrderStatus.PENDING,
                quantity=10,
                filled_quantity=0,
                price=None,
                filled_price=None,
                timestamp=datetime.now(UTC),
            )
        ],
    )

    reconcile_positions_and_orders(ctx)

    assert ctx.positions["AAPL"] == 10
    assert ctx.orders[0].status is OrderStatus.FILLED

