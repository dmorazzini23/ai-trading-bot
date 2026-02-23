from __future__ import annotations

from types import SimpleNamespace

from ai_trading.execution.engine import OrderManager


class _StubIntentStore:
    def __init__(self) -> None:
        self.closed: list[tuple[str, str]] = []

    def record_fill(self, intent_id: str, *, fill_qty: float, fill_price: float | None) -> None:
        del intent_id, fill_qty, fill_price

    def close_intent(self, intent_id: str, *, final_status: str) -> None:
        self.closed.append((intent_id, final_status))


def _manager_with_store() -> tuple[OrderManager, _StubIntentStore]:
    engine = OrderManager.__new__(OrderManager)
    store = _StubIntentStore()
    engine._intent_store = store
    engine._intent_by_order_id = {}
    engine._intent_reported_fill_qty = {}
    return engine, store


def test_broker_done_for_day_maps_to_expired_terminal() -> None:
    engine, store = _manager_with_store()
    order = SimpleNamespace(
        id="ord-1",
        status="done_for_day",
        filled_quantity=0.0,
        average_fill_price=None,
    )
    engine._intent_by_order_id["ord-1"] = "intent-1"

    engine._sync_intent_with_order_event(order, "updated")

    assert store.closed == [("intent-1", "EXPIRED")]


def test_broker_replaced_maps_to_closed_terminal() -> None:
    engine, store = _manager_with_store()
    order = SimpleNamespace(
        id="ord-2",
        status="replaced",
        filled_quantity=0.0,
        average_fill_price=None,
    )
    engine._intent_by_order_id["ord-2"] = "intent-2"

    engine._sync_intent_with_order_event(order, "updated")

    assert store.closed == [("intent-2", "CLOSED")]


def test_broker_stopped_maps_to_closed_terminal() -> None:
    engine, store = _manager_with_store()
    order = SimpleNamespace(
        id="ord-3",
        status="stopped",
        filled_quantity=0.0,
        average_fill_price=None,
    )
    engine._intent_by_order_id["ord-3"] = "intent-3"

    engine._sync_intent_with_order_event(order, "updated")

    assert store.closed == [("intent-3", "CLOSED")]
