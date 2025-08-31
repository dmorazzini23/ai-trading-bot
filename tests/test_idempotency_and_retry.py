from __future__ import annotations

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

