from __future__ import annotations

import pytest

from ai_trading.execution.engine import OrderManager
from ai_trading.oms.intent_store import IntentStore

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
    tmp_path,
) -> None:
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
