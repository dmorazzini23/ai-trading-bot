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

