from __future__ import annotations

from ai_trading.broker.adapters import (
    AlpacaBrokerAdapter,
    PaperBrokerAdapter,
    TradierBrokerAdapter,
    build_broker_adapter,
)


class _DummyClient:
    def __init__(self) -> None:
        self.submit_calls = 0
        self.orders = [{"id": "1", "status": "open"}]

    def get_account(self):
        return {"buying_power": "1234"}

    def list_orders(self, status: str = "open"):
        assert status == "open"
        return list(self.orders)

    def submit_order(self, order_data):
        self.submit_calls += 1
        return {"id": "dummy", "status": "accepted", "payload": dict(order_data)}


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
    result = adapter.submit_order({"symbol": "AAPL", "quantity": 1})

    assert account == {"buying_power": "1234"}
    assert orders == [{"id": "1", "status": "open"}]
    assert result["status"] == "accepted"
    assert client.submit_calls == 1
    _assert_submit_contract(result)


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
