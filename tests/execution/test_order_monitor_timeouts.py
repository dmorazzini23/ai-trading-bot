from __future__ import annotations

from datetime import UTC, datetime

from ai_trading.core.enums import OrderSide, OrderStatus, OrderType
from ai_trading.execution.engine import Order, OrderManager


def test_monitor_orders_uses_monotonic_for_expiry(monkeypatch):
    """Orders expire even when wall-clock time is frozen."""

    manager = OrderManager()
    manager.order_timeout = 1

    order = Order("SPY", OrderSide.BUY, 10, order_type=OrderType.MARKET)
    order._created_monotonic = 0.0
    order.created_at = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)

    manager.active_orders[order.id] = order

    events: list[tuple[str, str]] = []
    manager.execution_callbacks.append(lambda o, ev: events.append((o.id, ev)))

    freeze_ts = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)
    monkeypatch.setattr("ai_trading.execution.engine.safe_utcnow", lambda: freeze_ts)
    monkeypatch.setattr("ai_trading.execution.engine.monotonic_time", lambda: 2.0)
    monkeypatch.setattr(
        "ai_trading.execution.reconcile.reconcile_positions_and_orders", lambda *args, **kwargs: None
    )

    manager._monitor_orders_tick()

    assert order.id not in manager.active_orders
    assert order.status == OrderStatus.EXPIRED
    assert (order.id, "expired") in events
