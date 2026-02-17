from __future__ import annotations

from ai_trading.broker.adapters import (
    AlpacaBrokerAdapter,
    PaperBrokerAdapter,
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
    assert len(adapter.list_orders("open")) == 1


def test_build_broker_adapter_factory() -> None:
    dummy = _DummyClient()

    paper = build_broker_adapter(provider="paper", client=dummy)
    alpaca = build_broker_adapter(provider="alpaca", client=dummy)
    missing = build_broker_adapter(provider="alpaca", client=None)

    assert isinstance(paper, PaperBrokerAdapter)
    assert isinstance(alpaca, AlpacaBrokerAdapter)
    assert missing is None
