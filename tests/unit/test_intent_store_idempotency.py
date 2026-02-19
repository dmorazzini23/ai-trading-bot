from __future__ import annotations

from ai_trading.core.enums import OrderSide
from ai_trading.execution.idempotency import OrderIdempotencyCache
from ai_trading.oms.intent_store import IntentStore


def test_intent_store_enforces_unique_idempotency_key(tmp_path) -> None:
    store = IntentStore(path=str(tmp_path / "oms_intents.db"))
    first, created_first = store.create_intent(
        intent_id="intent-1",
        idempotency_key="k-abc",
        symbol="AAPL",
        side="buy",
        quantity=10,
    )
    second, created_second = store.create_intent(
        intent_id="intent-2",
        idempotency_key="k-abc",
        symbol="AAPL",
        side="buy",
        quantity=10,
    )
    assert created_first is True
    assert created_second is False
    assert second.intent_id == first.intent_id

    assert store.claim_for_submit(first.intent_id, stale_after_seconds=60) is True
    assert store.claim_for_submit(first.intent_id, stale_after_seconds=60) is False

    store.mark_submitted(first.intent_id, "broker-1")
    store.record_fill(first.intent_id, fill_qty=4, fill_price=190.25)
    fills = store.list_fills(first.intent_id)
    assert len(fills) == 1
    assert fills[0].fill_qty == 4
    assert fills[0].fill_price == 190.25

    store.close_intent(first.intent_id, final_status="FILLED")
    open_intents = store.get_open_intents()
    assert all(intent.intent_id != first.intent_id for intent in open_intents)


def test_idempotency_cache_uses_intent_store_for_restart_safe_dedup(tmp_path) -> None:
    store = IntentStore(path=str(tmp_path / "idempotency.db"))
    cache = OrderIdempotencyCache(ttl_seconds=60, max_size=100, intent_store=store)
    key = cache.generate_key("AAPL", OrderSide.BUY, 5.0)

    is_dup_first, existing_first = cache.check_and_mark_submitted(key, "order-1")
    is_dup_second, existing_second = cache.check_and_mark_submitted(key, "order-2")

    assert is_dup_first is False
    assert existing_first is None
    assert is_dup_second is True
    assert existing_second == "order-1"

    persisted = store.get_intent_by_key(key.hash())
    assert persisted is not None
    assert persisted.broker_order_id == "order-1"

