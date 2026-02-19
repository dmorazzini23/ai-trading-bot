from __future__ import annotations

from ai_trading.core.enums import OrderSide, OrderType
from ai_trading.execution.engine import Order, OrderManager
from ai_trading.oms.intent_store import IntentStore


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
        price=190.0,
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
        price=190.0,
    )
    duplicate_response = manager.submit_order(duplicate_order)
    assert duplicate_response is None


def test_reconcile_marks_missing_submitted_intent_failed(tmp_path) -> None:
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
    assert summary["marked_failed"] == 1

    refreshed = store.get_intent(intent.intent_id)
    assert refreshed is not None
    assert refreshed.status == "FAILED"
    assert refreshed.last_error == "reconcile_missing_broker_order"


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
