from __future__ import annotations

import pytest

from ai_trading.execution.engine import OrderManager
from ai_trading.oms.event_store import EventStore
from ai_trading.oms.intent_store import IntentStore
from ai_trading.oms.invariants import evaluate_oms_lifecycle_parity_invariants

pytest.importorskip("sqlalchemy")


@pytest.mark.parametrize("initial_status", ["SUBMITTED", "PARTIALLY_FILLED"])
def test_reconcile_missing_open_orders_does_not_fail_open_intent(
    tmp_path,
    initial_status: str,
) -> None:
    store = IntentStore(path=str(tmp_path / "reconcile_missing_open_snapshot.db"))
    manager = OrderManager()
    manager.configure_intent_store(store)

    intent, created = store.create_intent(
        intent_id=f"intent-missing-open-snapshot-{initial_status.lower()}",
        idempotency_key=f"missing-open-snapshot-{initial_status.lower()}",
        symbol="MSFT",
        side="buy",
        quantity=5.0,
        status="SUBMITTED",
    )
    assert created is True
    store.mark_submitted(intent.intent_id, "broker-order-404")
    if initial_status == "PARTIALLY_FILLED":
        store.record_fill(intent.intent_id, fill_qty=1.0, fill_price=190.25)

    summary = manager.reconcile_open_intents(broker_orders=[])

    assert summary["intents_checked"] == 1
    assert summary["marked_failed"] == 0

    refreshed = store.get_intent(intent.intent_id)
    assert refreshed is not None
    assert refreshed.status == initial_status
    assert refreshed.last_error in (None, "")

    open_intent_ids = {record.intent_id for record in store.get_open_intents()}
    assert intent.intent_id in open_intent_ids


def test_reconcile_missing_open_orders_closes_intent_on_terminal_broker_lookup(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_DUAL_WRITE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_JSONL_ENABLED", "0")
    store = IntentStore(path=str(tmp_path / "reconcile_terminal_lookup.db"))
    manager = OrderManager()
    manager.configure_intent_store(store)

    intent, created = store.create_intent(
        intent_id="intent-terminal-lookup",
        idempotency_key="terminal-lookup-key",
        symbol="MSFT",
        side="buy",
        quantity=5.0,
        status="SUBMITTED",
    )
    assert created is True
    store.mark_submitted(intent.intent_id, "broker-order-505")

    def get_order_by_id(order_id: str) -> dict[str, str]:
        assert order_id == "broker-order-505"
        return {
            "id": order_id,
            "client_order_id": intent.intent_id,
            "status": "filled",
        }

    manager.reconcile_open_intents(
        broker_orders=[],
        get_order_by_id_fn=get_order_by_id,
    )

    refreshed = store.get_intent(intent.intent_id)
    assert refreshed is not None
    assert refreshed.status == "FILLED"
    open_intent_ids = {record.intent_id for record in store.get_open_intents()}
    assert intent.intent_id not in open_intent_ids

    event_store = EventStore(path=str(tmp_path / "reconcile_terminal_lookup.db"))
    rows = event_store.list_oms_events(intent_id=intent.intent_id, limit=5000)
    event_store.close()
    event_types = [str(row.get("event_type") or "").strip().upper() for row in rows]
    assert "ORDER_PARTIALLY_FILLED" in event_types
    assert "ORDER_FILLED" in event_types
    assert "INTENT_CLOSED" in event_types

    parity = evaluate_oms_lifecycle_parity_invariants(
        intent_store_path=str(tmp_path / "reconcile_terminal_lookup.db")
    )
    assert int(parity["violations"]["filled_missing_partial_fill"]) == 0


def test_reconcile_terminal_lookup_maps_done_for_day_to_expired(
    tmp_path,
) -> None:
    store = IntentStore(path=str(tmp_path / "reconcile_terminal_done_for_day.db"))
    manager = OrderManager()
    manager.configure_intent_store(store)

    intent, created = store.create_intent(
        intent_id="intent-terminal-done-for-day",
        idempotency_key="terminal-done-for-day-key",
        symbol="MSFT",
        side="buy",
        quantity=5.0,
        status="SUBMITTED",
    )
    assert created is True
    store.mark_submitted(intent.intent_id, "broker-order-606")

    def get_order_by_id(order_id: str) -> dict[str, str]:
        assert order_id == "broker-order-606"
        return {
            "id": order_id,
            "client_order_id": intent.intent_id,
            "status": "done_for_day",
        }

    manager.reconcile_open_intents(
        broker_orders=[],
        get_order_by_id_fn=get_order_by_id,
    )

    refreshed = store.get_intent(intent.intent_id)
    assert refreshed is not None
    assert refreshed.status == "EXPIRED"
    open_intent_ids = {record.intent_id for record in store.get_open_intents()}
    assert intent.intent_id not in open_intent_ids
