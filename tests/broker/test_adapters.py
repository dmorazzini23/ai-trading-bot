from __future__ import annotations

from typing import Any

from ai_trading.broker.adapters import (
    AlpacaBrokerAdapter,
    PaperBrokerAdapter,
    TradierBrokerAdapter,
    build_broker_adapter,
)
from ai_trading.oms.orders import build_bracket, build_oco, build_oto


class _DummyClient:
    def __init__(self) -> None:
        self.submit_calls = 0
        self.submitted_request: Any | None = None
        self.orders = [{"id": "1", "status": "open"}]

    def get_account(self):
        return {"buying_power": "1234"}

    def list_orders(self, status: str = "open"):
        assert status == "open"
        return list(self.orders)

    def submit_order(self, order_data):
        self.submit_calls += 1
        self.submitted_request = order_data
        return {"id": "dummy", "status": "accepted", "payload": order_data}


class _NativeGetOrdersClient:
    def __init__(self) -> None:
        self.filter = None
        self.list_orders_called = False

    def get_orders(self, *, filter):
        self.filter = filter
        return [{"id": "native-1", "status": "open"}]

    def list_orders(self, **_kwargs):
        self.list_orders_called = True
        raise AssertionError("native alpaca-py order listing must use get_orders(filter=...)")


class _FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http_{self.status_code}")

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self._responses:
            raise AssertionError("missing fake response")
        return self._responses.pop(0)


def _assert_submit_contract(result) -> None:
    assert isinstance(result, dict)
    assert str(result.get("id", "")).strip()
    assert str(result.get("status", "")).strip()


def test_alpaca_adapter_passthrough() -> None:
    client = _DummyClient()
    adapter = AlpacaBrokerAdapter(client=client)

    account = adapter.get_account()
    orders = adapter.list_orders("open")
    result = adapter.submit_order({"symbol": "AAPL", "side": "buy", "quantity": 1})

    assert account == {"buying_power": "1234"}
    assert orders == [{"id": "1", "status": "open"}]
    assert result["status"] == "accepted"
    assert client.submit_calls == 1
    assert client.submitted_request is not None
    assert client.submitted_request.symbol == "AAPL"
    assert client.submitted_request.qty == 1
    _assert_submit_contract(result)


def test_alpaca_adapter_builds_native_bracket_request_from_mapping() -> None:
    client = _DummyClient()
    adapter = AlpacaBrokerAdapter(client=client)

    adapter.submit_order(
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 1,
            "type": "limit",
            "limit_price": 125.50,
            "order_class": "bracket",
            "take_profit": {"limit_price": 130.0},
            "stop_loss": {"stop_price": 120.0, "limit_price": 119.5},
        },
    )

    request = client.submitted_request
    assert request is not None
    assert request.limit_price == 125.50
    assert request.order_class.value == "bracket"
    assert request.take_profit.limit_price == 130.0
    assert request.stop_loss.stop_price == 120.0
    assert request.stop_loss.limit_price == 119.5


def test_alpaca_adapter_translates_legacy_order_family_legs() -> None:
    client = _DummyClient()
    adapter = AlpacaBrokerAdapter(client=client)

    adapter.submit_order(
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 1,
            "type": "bracket",
            "limit_price": 125.50,
            "legs": {
                "take_profit": {"limit_price": 130.0},
                "stop_loss": {"stop_price": 120.0},
            },
        },
    )

    request = client.submitted_request
    assert request.type.value == "limit"
    assert request.order_class.value == "bracket"
    assert request.take_profit.limit_price == 130.0
    assert request.stop_loss.stop_price == 120.0


def test_oms_order_family_builders_are_alpaca_adapter_compatible() -> None:
    bracket = build_bracket(
        symbol="AAPL",
        side="buy",
        qty=1,
        entry_limit=100,
        take_profit=110,
        stop_loss=95,
    )
    oco = build_oco(symbol="AAPL", side="sell", qty=1, take_profit=110, stop_loss=95)
    oto = build_oto(symbol="AAPL", side="buy", qty=1, entry_limit=100, stop_loss=95)

    for payload, order_class in ((bracket, "bracket"), (oco, "oco"), (oto, "oto")):
        client = _DummyClient()
        AlpacaBrokerAdapter(client=client).submit_order(payload)
        request = client.submitted_request
        assert request.type.value == "limit"
        assert request.order_class.value == order_class
        assert request.stop_loss.stop_price == 95.0


def test_alpaca_adapter_preserves_short_position_intent() -> None:
    client = _DummyClient()
    adapter = AlpacaBrokerAdapter(client=client)

    adapter.submit_order({"symbol": "MSFT", "side": "sell_short", "quantity": 2})

    assert client.submitted_request.side.value == "sell"
    assert client.submitted_request.position_intent.value == "sell_to_open"

    adapter.submit_order({"symbol": "MSFT", "side": "buy_to_cover", "quantity": 2})

    assert client.submitted_request.side.value == "buy"
    assert client.submitted_request.position_intent.value == "buy_to_close"


def test_alpaca_adapter_uses_native_get_orders_filter() -> None:
    client = _NativeGetOrdersClient()
    adapter = AlpacaBrokerAdapter(client=client)

    orders = adapter.list_orders("open")

    assert orders == [{"id": "native-1", "status": "open"}]
    assert client.list_orders_called is False
    assert client.filter is not None
    assert getattr(client.filter, "status", None).value == "open"


def test_alpaca_adapter_rejects_unknown_side() -> None:
    adapter = AlpacaBrokerAdapter(client=_DummyClient())

    try:
        adapter.submit_order({"symbol": "AAPL", "side": "hold", "quantity": 1})
    except ValueError as exc:
        assert "Unsupported broker order side" in str(exc)
    else:  # pragma: no cover - assertion aid
        raise AssertionError("unknown side should fail closed")


def test_paper_adapter_supports_account_orders_and_submit() -> None:
    adapter = PaperBrokerAdapter()

    account = adapter.get_account()
    assert account["buying_power"] == "100000"
    assert adapter.list_orders("open") == []

    response = adapter.submit_order(
        {"symbol": "MSFT", "side": "buy", "quantity": 2, "limit_price": 300.0},
    )

    assert response["status"] == "accepted"
    assert response["symbol"] == "MSFT"
    assert response["qty"] == 2
    assert response["quantity"] == 2
    assert response["filled_qty"] == "0"
    assert response["filled_quantity"] == "0"
    assert response["filled_avg_price"] is None
    assert response["type"] == "market"
    assert response["time_in_force"] == "day"
    assert len(adapter.list_orders("open")) == 1
    _assert_submit_contract(response)


def test_paper_adapter_normalizes_short_side_alias() -> None:
    adapter = PaperBrokerAdapter()

    response = adapter.submit_order({"symbol": "MSFT", "side": "short", "quantity": 2})

    assert response["side"] == "sell_short"
    assert adapter.list_orders("open")[0]["side"] == "sell_short"


def test_tradier_adapter_contract_parity() -> None:
    session = _FakeSession(
        responses=[
            _FakeResponse({"balances": {"cash": "5000", "buying_power": "10000"}}),
            _FakeResponse({"orders": {"order": {"id": "ord-1", "status": "open", "symbol": "AAPL"}}}),
            _FakeResponse({"order": {"id": "ord-2", "status": "submitted"}}),
        ],
    )
    adapter = TradierBrokerAdapter(
        token="token-123",
        account_id="acct-1",
        base_url="https://sandbox.tradier.com/v1",
        session=session,
    )

    account = adapter.get_account()
    orders = adapter.list_orders("open")
    submitted = adapter.submit_order(
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 2,
            "type": "limit",
            "limit_price": 100.5,
            "time_in_force": "day",
            "client_order_id": "cid-1",
        },
    )

    assert account["buying_power"] == "10000"
    assert len(orders) == 1
    assert orders[0]["id"] == "ord-1"
    assert submitted["id"] == "ord-2"
    assert submitted["status"] == "submitted"
    assert submitted["client_order_id"] == "cid-1"
    _assert_submit_contract(submitted)

    assert len(session.calls) == 3
    assert session.calls[0]["url"].endswith("/accounts/acct-1/balances")
    assert session.calls[1]["url"].endswith("/accounts/acct-1/orders")
    assert session.calls[2]["url"].endswith("/accounts/acct-1/orders")
    third_data = session.calls[2]["data"]
    assert third_data["class"] == "equity"
    assert third_data["type"] == "limit"
    assert third_data["price"] == "100.5"
    assert third_data["tag"] == "cid-1"


def test_tradier_adapter_normalizes_list_order_aliases() -> None:
    session = _FakeSession(
        responses=[
            _FakeResponse(
                {
                    "orders": {
                        "order": {
                            "id": 228175,
                            "status": "partially_filled",
                            "symbol": "aapl",
                            "side": "short",
                            "quantity": "10.00000000",
                            "exec_quantity": "3.00000000",
                            "avg_fill_price": "175.25",
                            "tag": "cid-tradier-1",
                        }
                    }
                }
            ),
        ],
    )
    adapter = TradierBrokerAdapter(
        token="token-123",
        account_id="acct-1",
        base_url="https://sandbox.tradier.com/v1",
        session=session,
    )

    orders = adapter.list_orders("open")

    assert orders == [
        {
            "id": "228175",
            "status": "partially_filled",
            "symbol": "AAPL",
            "side": "sell_short",
            "quantity": "10.00000000",
            "exec_quantity": "3.00000000",
            "avg_fill_price": "175.25",
            "tag": "cid-tradier-1",
            "qty": "10.00000000",
            "filled_qty": "3.00000000",
            "filled_quantity": "3.00000000",
            "filled_avg_price": "175.25",
            "client_order_id": "cid-tradier-1",
        }
    ]


def test_tradier_adapter_normalizes_submit_fill_aliases_and_terminal_error() -> None:
    session = _FakeSession(
        responses=[
            _FakeResponse(
                {
                    "order": {
                        "id": "ord-error-1",
                        "status": "error",
                        "exec_quantity": "2",
                        "avg_fill_price": "101.25",
                        "tag": "cid-error-1",
                    }
                }
            ),
        ],
    )
    adapter = TradierBrokerAdapter(
        token="token-123",
        account_id="acct-1",
        base_url="https://sandbox.tradier.com/v1",
        session=session,
    )

    submitted = adapter.submit_order(
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 5,
            "client_order_id": "cid-error-1",
        },
    )

    assert submitted["id"] == "ord-error-1"
    assert submitted["status"] == "rejected"
    assert submitted["client_order_id"] == "cid-error-1"
    assert submitted["qty"] == 5
    assert submitted["quantity"] == 5
    assert submitted["filled_qty"] == "2"
    assert submitted["filled_quantity"] == "2"
    assert submitted["filled_avg_price"] == "101.25"
    assert session.calls[0]["data"]["tag"] == "cid-error-1"


def test_tradier_adapter_normalizes_short_side_alias() -> None:
    session = _FakeSession(
        responses=[
            _FakeResponse({"order": {"id": "ord-short-1", "status": "submitted"}}),
        ],
    )
    adapter = TradierBrokerAdapter(
        token="token-123",
        account_id="acct-1",
        base_url="https://sandbox.tradier.com/v1",
        session=session,
    )

    submitted = adapter.submit_order(
        {
            "symbol": "TSLA",
            "side": "short",
            "quantity": 3,
            "type": "market",
        },
    )

    assert submitted["side"] == "sell_short"
    assert session.calls[0]["data"]["side"] == "sell_short"


def test_build_broker_adapter_factory(monkeypatch) -> None:
    dummy = _DummyClient()

    paper = build_broker_adapter(provider="paper", client=dummy)
    alpaca = build_broker_adapter(provider="alpaca", client=dummy)
    missing = build_broker_adapter(provider="alpaca", client=None)
    monkeypatch.setenv("TRADIER_ACCESS_TOKEN", "token-x")
    monkeypatch.setenv("TRADIER_ACCOUNT_ID", "acct-x")
    tradier = build_broker_adapter(provider="tradier", client=_FakeSession([]))

    assert isinstance(paper, PaperBrokerAdapter)
    assert isinstance(alpaca, AlpacaBrokerAdapter)
    assert isinstance(tradier, TradierBrokerAdapter)
    assert missing is None


def test_build_broker_adapter_tradier_requires_credentials(monkeypatch) -> None:
    monkeypatch.delenv("TRADIER_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("TRADIER_ACCOUNT_ID", raising=False)

    adapter = build_broker_adapter(provider="tradier", client=_FakeSession([]))

    assert adapter is None
