import pytest

from ai_trading.broker.alpaca import AlpacaBroker

pytestmark = pytest.mark.alpaca

# Fake “new” and “old” clients that expose just enough to be called.


class FakeNew:
    def __init__(self):
        self.mode = "new"
        self.orders = []

    # simulate new methods
    def get_orders(self, req):  # req is ignored in fake
        return ["o-new-open"]

    def cancel_orders(self):
        return "ok-new-cancel-all"

    def cancel_order_by_id(self, oid):
        return f"ok-new-cancel:{oid}"

    def get_all_positions(self):
        return ["pos-new"]

    def get_account(self):
        return {"equity": "100.0"}

    def submit_order(self, req):
        return {"submitted_via": "new"}


class FakeOld:
    def __init__(self):
        self.mode = "old"

    # simulate old methods
    def list_orders(self, status="all", limit=None):
        return [f"o-old-{status}"]

    def cancel_all_orders(self):
        return "ok-old-cancel-all"

    def cancel_order(self, oid):
        return f"ok-old-cancel:{oid}"

    def list_positions(self):
        return ["pos-old"]

    def get_account(self):
        return {"equity": "200.0"}

    def submit_order(self, **kwargs):
        return {"submitted_via": "old", **kwargs}


def test_adapter_orders_new(monkeypatch):
    pytest.importorskip("alpaca.trading.client")
    fake = FakeNew()
    broker = AlpacaBroker(fake)
    broker._is_new = True
    class DummyReq:
        def __init__(self, *args, **kwargs):
            pass

    class DummyStatus:
        OPEN = object()

    broker._GetOrdersRequest = DummyReq
    broker._QueryOrderStatus = DummyStatus
    out = broker.list_open_orders()
    assert out == ["o-new-open"]


def test_adapter_orders_old():
    fake = FakeOld()
    broker = AlpacaBroker(fake)
    out = broker.list_open_orders()
    assert out == ["o-old-open"]


def test_positions_and_account_old():
    fake = FakeOld()
    b = AlpacaBroker(fake)
    assert b.list_open_positions() == ["pos-old"]
    assert b.get_account()["equity"] == "200.0"
