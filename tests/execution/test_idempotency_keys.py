from datetime import UTC, datetime

from ai_trading.execution.idempotency import OrderIdempotencyCache
from ai_trading.order.types import OrderSide


def test_idempotency_key_includes_order_intent_fields() -> None:
    cache = OrderIdempotencyCache()
    ts = datetime(2026, 4, 29, 12, 0, tzinfo=UTC)

    limit_key = cache.generate_key(
        "AAPL",
        OrderSide.BUY,
        1,
        ts,
        client_order_id="cid-1",
        order_type="limit",
        limit_price="100.00",
    )
    market_key = cache.generate_key(
        "AAPL",
        OrderSide.BUY,
        1,
        ts,
        client_order_id="cid-1",
        order_type="market",
    )
    other_client_key = cache.generate_key(
        "AAPL",
        OrderSide.BUY,
        1,
        ts,
        client_order_id="cid-2",
        order_type="limit",
        limit_price="100.00",
    )

    assert limit_key.hash() != market_key.hash()
    assert limit_key.hash() != other_client_key.hash()
