from __future__ import annotations

import threading
import types
from datetime import UTC, datetime


def test_order_idempotency_duplicate_prevented():
    from ai_trading.execution.engine import OrderManager, Order
    from ai_trading.core.enums import OrderSide, OrderType

    om = OrderManager()
    o1 = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)
    resp1 = om.submit_order(o1)
    assert resp1 is not None, "first submit should be accepted"

    # Duplicate order with the same key parameters (symbol/side/qty)
    o2 = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)
    resp2 = om.submit_order(o2)

    # Duplicate should be rejected by idempotency cache
    assert resp2 is None
    assert getattr(o2, "status", None) is not None


def test_idempotency_check_and_mark_is_atomic_across_threads():
    from ai_trading.execution.idempotency import OrderIdempotencyCache

    cache = OrderIdempotencyCache(ttl_seconds=60, max_size=100)
    key = cache.generate_key("AAPL", "buy", 5)
    barrier = threading.Barrier(2)
    results: list[tuple[bool, str | None]] = []
    errors: list[Exception] = []

    def _worker(order_id: str) -> None:
        try:
            barrier.wait(timeout=2)
            results.append(cache.check_and_mark_submitted(key, order_id))
        except Exception as exc:  # pragma: no cover - defensive capture
            errors.append(exc)

    t1 = threading.Thread(target=_worker, args=("order-1",))
    t2 = threading.Thread(target=_worker, args=("order-2",))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert not errors
    assert len(results) == 2
    assert sum(1 for is_dup, _existing in results if is_dup) == 1
    assert sum(1 for is_dup, _existing in results if not is_dup) == 1


def test_order_manager_submit_order_uses_atomic_idempotency_check(monkeypatch):
    from ai_trading.execution.engine import OrderManager, Order
    from ai_trading.core.enums import OrderSide, OrderType

    manager = OrderManager()
    cache = manager._ensure_idempotency_cache()

    def _fail_if_called(_key):
        raise AssertionError("legacy non-atomic idempotency path should not be used")

    monkeypatch.setattr(cache, "is_duplicate", _fail_if_called)

    order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=1, order_type=OrderType.MARKET)
    response = manager.submit_order(order)

    assert response is not None


def test_order_manager_logs_submit_skipped_for_duplicate(caplog):
    from ai_trading.execution.engine import OrderManager, Order
    from ai_trading.core.enums import OrderSide, OrderType

    manager = OrderManager()
    first = Order(symbol="AAPL", side=OrderSide.BUY, quantity=25, order_type=OrderType.MARKET)
    duplicate = Order(symbol="AAPL", side=OrderSide.BUY, quantity=25, order_type=OrderType.MARKET)

    with caplog.at_level("INFO"):
        assert manager.submit_order(first) is not None
        assert manager.submit_order(duplicate) is None

    skipped = [record for record in caplog.records if record.msg == "ORDER_SUBMIT_SKIPPED"]
    assert skipped
    assert any(getattr(record, "reason", None) == "duplicate_order" for record in skipped)


def test_order_manager_logs_submit_skipped_for_validation_failure(caplog):
    from ai_trading.execution.engine import OrderManager, Order
    from ai_trading.core.enums import OrderSide, OrderType

    manager = OrderManager()
    invalid = Order(symbol="AAPL", side=OrderSide.BUY, quantity=0, order_type=OrderType.MARKET)

    with caplog.at_level("INFO"):
        assert manager.submit_order(invalid) is None

    skipped = [record for record in caplog.records if record.msg == "ORDER_SUBMIT_SKIPPED"]
    assert skipped
    assert any(getattr(record, "reason", None) == "validation_failed" for record in skipped)


def test_http_submit_retries_once_then_succeeds(monkeypatch):
    """Simulate transient network error on first call, success on second.

    Verifies retry wrapper in HTTP submit path without real network.
    """
    from ai_trading.alpaca_api import _http_submit, _AlpacaConfig
    from ai_trading.exc import RequestException
    from ai_trading.alpaca_api import _HTTP

    calls = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
        calls["n"] += 1
        if calls["n"] == 1:
            raise RequestException("transient error")

        class Resp:
            status_code = 200

            def json(self):
                return {
                    "id": "test-oid",
                    "client_order_id": "idem-1",
                    "symbol": json.get("symbol"),  # type: ignore[union-attr]
                    "qty": json.get("qty"),
                    "side": json.get("side"),
                    "type": json.get("type"),
                    "time_in_force": json.get("time_in_force"),
                    "status": "accepted",
                    "submitted_at": datetime.now(UTC).isoformat(),
                    "filled_qty": "0",
                }

        return Resp()

    monkeypatch.setattr(_HTTP, "post", fake_post)

    cfg = _AlpacaConfig(base_url="https://paper-api.alpaca.markets", key_id="k", secret_key="s", shadow=False)
    result = _http_submit(
        cfg,
        symbol="MSFT",
        qty=10,
        side="buy",
        type="market",
        time_in_force="day",
        limit_price=None,
        stop_price=None,
        idempotency_key="idem-1",
        timeout=2,
    )
    assert result["symbol"] == "MSFT"
    assert calls["n"] == 2, "should have retried once"
