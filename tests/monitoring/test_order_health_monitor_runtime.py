from __future__ import annotations

import json
import time
from datetime import timedelta
from collections.abc import Generator
from pathlib import Path

import pytest

from ai_trading.monitoring import order_health_monitor as ohm
from ai_trading.utils.time import safe_utcnow


@pytest.fixture(autouse=True)
def _clear_order_monitor_state() -> Generator[None, None, None]:
    with ohm._order_tracking_lock:
        ohm._active_orders.clear()
    ohm._order_health_monitor = None
    yield
    with ohm._order_tracking_lock:
        ohm._active_orders.clear()
    ohm._order_health_monitor = None


class _ExecutionEngine:
    def __init__(self) -> None:
        self.canceled: list[str] = []

    def _cancel_stale_order(self, order_id: str) -> bool:
        self.canceled.append(order_id)
        return True


def test_order_info_quantity_alias() -> None:
    order = ohm.OrderInfo(
        order_id="order-1",
        symbol="AAPL",
        side="buy",
        qty=17,
        submitted_time=time.time(),
        last_status="NEW",
    )

    assert order.quantity == 17




























