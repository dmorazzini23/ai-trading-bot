from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from ai_trading.core.interfaces import Order, OrderStatus, OrderType
from ai_trading.execution.reconcile import reconcile_positions_and_orders
from ai_trading.order.types import OrderSide


class _Broker:
    def get_order(self, _order_id: str) -> SimpleNamespace:
        return SimpleNamespace(status="filled", filled_qty="10")

    def list_positions(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(symbol="AAPL", qty="10")]

    def list_orders(self, *, status: str) -> list[SimpleNamespace]:
        assert status == "open"
        return []


def test_reconcile_applies_only_new_fill_delta_to_local_position() -> None:
    order = Order(
        id="ord-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        status=OrderStatus.PARTIALLY_FILLED,
        quantity=10,
        filled_quantity=5,
        price=None,
        filled_price=None,
        timestamp=datetime(2026, 4, 26, tzinfo=UTC),
    )
    ctx = SimpleNamespace(api=_Broker(), positions={"AAPL": 5}, orders=[order])

    result = reconcile_positions_and_orders(ctx)

    assert result.position_drifts == []
    assert ctx.positions["AAPL"] == 10
    assert ctx.orders[0].filled_quantity == 10
