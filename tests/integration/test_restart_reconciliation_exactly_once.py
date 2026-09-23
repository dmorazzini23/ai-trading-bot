from __future__ import annotations

from typing import Any, cast

import pytest
from sqlalchemy import text

from ai_trading.core.enums import OrderSide, OrderType
from ai_trading.execution.engine import Order, OrderManager
from ai_trading.oms.intent_store import IntentStore

pytest.importorskip("sqlalchemy")


def test_restart_like_pending_intent_submits_once(tmp_path) -> None:
    store = IntentStore(path=str(tmp_path / "restart_intents.db"))
    manager = OrderManager()
    manager.configure_intent_store(store)
    cache = manager._ensure_idempotency_cache()

    key = cache.generate_key("AAPL", OrderSide.BUY, 15)
    seeded, created = store.create_intent(
        intent_id="intent-before-crash",
        idempotency_key=key.hash(),
        symbol="AAPL",
        side="buy",
        quantity=15.0,
        status="PENDING_SUBMIT",
    )
    assert created is True
    assert seeded.intent_id == "intent-before-crash"

    first_order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=15,
        order_type=OrderType.LIMIT,
        price=cast(Any, 190.0),
    )
    first_response = manager.submit_order(first_order)
    assert first_response is not None

    persisted = store.get_intent_by_key(key.hash())
    assert persisted is not None
    assert persisted.intent_id == "intent-before-crash"
    assert persisted.broker_order_id == first_order.id
    assert persisted.status == "SUBMITTED"

    duplicate_order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=15,
        order_type=OrderType.LIMIT,
        price=cast(Any, 190.0),
    )
    duplicate_response = manager.submit_order(duplicate_order)
    assert duplicate_response is None


def test_reconcile_keeps_missing_submitted_intent_open(tmp_path) -> None:
    store = IntentStore(path=str(tmp_path / "reconcile_missing.db"))
    manager = OrderManager()
    manager.configure_intent_store(store)

    intent, created = store.create_intent(
        intent_id="intent-missing-broker",
        idempotency_key="missing-broker-key",
        symbol="MSFT",
        side="buy",
        quantity=5.0,
        status="SUBMITTED",
    )
    assert created is True
    store.mark_submitted(intent.intent_id, "broker-order-404")

    summary = manager.reconcile_open_intents(broker_orders=[])
    assert summary["intents_checked"] == 1
    assert summary["marked_failed"] == 0

    refreshed = store.get_intent(intent.intent_id)
    assert refreshed is not None
    assert refreshed.status == "SUBMITTED"
    assert refreshed.last_error in (None, "")
    open_intent_ids = {record.intent_id for record in store.get_open_intents()}
    assert intent.intent_id in open_intent_ids


def test_reconcile_links_broker_id_from_client_order_id(tmp_path) -> None:
    store = IntentStore(path=str(tmp_path / "reconcile_client_id.db"))
    manager = OrderManager()
    manager.configure_intent_store(store)

    intent, created = store.create_intent(
        intent_id="intent-client-link",
        idempotency_key="intent-client-link-key",
        symbol="NVDA",
        side="buy",
        quantity=2.0,
        status="SUBMITTED",
    )
    assert created is True

    broker_orders = [
        {
            "id": "broker-order-200",
            "client_order_id": "intent-client-link",
            "status": "open",
            "symbol": "NVDA",
        }
    ]
    summary = manager.reconcile_open_intents(broker_orders=broker_orders)
    assert summary["intents_checked"] == 1
    assert summary["matched_open_orders"] == 1
    assert summary["marked_submitted"] == 1
    assert summary["marked_failed"] == 0

    refreshed = store.get_intent(intent.intent_id)
    assert refreshed is not None
    assert refreshed.status == "SUBMITTED"
    assert refreshed.broker_order_id == "broker-order-200"


def test_stale_submitting_intent_cannot_be_claimed_again(tmp_path) -> None:
    store = IntentStore(path=str(tmp_path / "stale_claim.db"))
    intent, created = store.create_intent(
        intent_id="client-accepted-response-lost",
        idempotency_key="one-broker-attempt",
        symbol="AAPL",
        side="buy",
        quantity=1.0,
        status="PENDING_SUBMIT",
    )
    assert created
    assert store.claim_for_submit(intent.intent_id, stale_after_seconds=1)
    with store._engine.begin() as connection:
        connection.execute(
            text("UPDATE intents SET updated_at = :past WHERE intent_id = :intent_id"),
            {"past": "2020-01-01T00:00:00+00:00", "intent_id": intent.intent_id},
        )

    assert not store.claim_for_submit(intent.intent_id, stale_after_seconds=1)
    refreshed = store.get_intent(intent.intent_id)
    assert refreshed is not None
    assert refreshed.status == "SUBMITTING"
    assert refreshed.submit_attempts == 1


def test_reconcile_lost_ack_finds_terminal_broker_order(tmp_path) -> None:
    store = IntentStore(path=str(tmp_path / "lost_ack.db"))
    manager = OrderManager()
    manager.configure_intent_store(store)
    intent, created = store.create_intent(
        intent_id="client-filled-after-disconnect",
        idempotency_key="lost-ack-broker-lookup",
        symbol="AAPL",
        side="buy",
        quantity=2.0,
        status="PENDING_SUBMIT",
    )
    assert created
    assert store.claim_for_submit(intent.intent_id)

    summary = manager.reconcile_open_intents(
        broker_orders=[],
        get_order_by_client_order_id_fn=lambda client_id: {
            "id": "broker-filled-1",
            "client_order_id": client_id,
            "status": "filled",
            "filled_qty": "2",
            "filled_avg_price": "100.25",
        },
    )

    refreshed = store.get_intent(intent.intent_id)
    assert refreshed is not None
    assert refreshed.status == "FILLED"
    assert refreshed.broker_order_id == "broker-filled-1"
    assert summary["marked_failed"] == 0


def test_reconcile_lost_ack_links_accepted_broker_order(tmp_path) -> None:
    store = IntentStore(path=str(tmp_path / "lost_ack_accepted.db"))
    manager = OrderManager()
    manager.configure_intent_store(store)
    intent, created = store.create_intent(
        intent_id="client-accepted-no-ack",
        idempotency_key="accepted-no-ack-key",
        symbol="AAPL",
        side="buy",
        quantity=2.0,
        status="PENDING_SUBMIT",
    )
    assert created
    assert store.claim_for_submit(intent.intent_id)

    summary = manager.reconcile_open_intents(
        broker_orders=[],
        get_order_by_client_order_id_fn=lambda client_id: {
            "id": "broker-accepted-1",
            "client_order_id": client_id,
            "status": "accepted",
            "filled_qty": "0",
        },
    )

    refreshed = store.get_intent(intent.intent_id)
    assert refreshed is not None
    assert refreshed.status == "SUBMITTED"
    assert refreshed.broker_order_id == "broker-accepted-1"
    assert summary["marked_submitted"] == 1


def test_reconcile_partial_fill_during_disconnect(tmp_path) -> None:
    store = IntentStore(path=str(tmp_path / "partial_disconnect.db"))
    manager = OrderManager()
    manager.configure_intent_store(store)
    intent, created = store.create_intent(
        intent_id="client-partial-no-ack",
        idempotency_key="partial-no-ack-key",
        symbol="AAPL",
        side="buy",
        quantity=3.0,
        status="PENDING_SUBMIT",
    )
    assert created
    assert store.claim_for_submit(intent.intent_id)
    broker_order = {
        "id": "broker-partial-1",
        "client_order_id": intent.intent_id,
        "status": "partially_filled",
        "filled_qty": "1",
        "filled_avg_price": "100.25",
    }

    first = manager.reconcile_open_intents(
        broker_orders=[],
        get_order_by_client_order_id_fn=lambda _client_id: broker_order,
    )
    second = manager.reconcile_open_intents(
        broker_orders=[],
        get_order_by_id_fn=lambda _order_id: broker_order,
    )

    refreshed = store.get_intent(intent.intent_id)
    assert refreshed is not None
    assert refreshed.status == "PARTIALLY_FILLED"
    assert refreshed.broker_order_id == "broker-partial-1"
    assert first["marked_submitted"] == 1
    assert second["marked_failed"] == 0
    fills = store.list_fills(intent.intent_id)
    assert len(fills) == 1
    assert fills[0].fill_qty == 1.0


def test_reconcile_lost_ack_rejects_mismatched_broker_identity(tmp_path) -> None:
    store = IntentStore(path=str(tmp_path / "lost_ack_mismatch.db"))
    manager = OrderManager()
    manager.configure_intent_store(store)
    intent, created = store.create_intent(
        intent_id="client-expected",
        idempotency_key="expected-key",
        symbol="AAPL",
        side="buy",
        quantity=1.0,
        status="PENDING_SUBMIT",
    )
    assert created
    assert store.claim_for_submit(intent.intent_id)

    summary = manager.reconcile_open_intents(
        broker_orders=[],
        get_order_by_client_order_id_fn=lambda _client_id: {
            "id": "broker-unrelated",
            "client_order_id": "client-other",
            "status": "filled",
            "filled_qty": "1",
        },
    )

    refreshed = store.get_intent(intent.intent_id)
    assert refreshed is not None
    assert refreshed.status == "SUBMITTING"
    assert summary["errors"] == 1
