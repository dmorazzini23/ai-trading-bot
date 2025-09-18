import types
import pytest

from ai_trading import alpaca_api  # AI-AGENT-REF: canonical import


class DummyAPI:
    def __init__(self):
        self.ids: list[str] = []

    def submit_order(self, **order_data):
        return types.SimpleNamespace(id=len(self.ids) + 1, **order_data)


def make_req(symbol="AAPL"):
    return types.SimpleNamespace(symbol=symbol, qty=1, side="buy", time_in_force="day")


@pytest.fixture
def api():
    api = DummyAPI()
    yield api
    api.ids.clear()
    if hasattr(api, "client_order_ids"):
        api.client_order_ids.clear()
    if hasattr(api, "client_order_ids") and not api.client_order_ids:
        delattr(api, "client_order_ids")


def test_unique_client_order_id(api):
    req1 = make_req()
    req2 = make_req()
    alpaca_api.submit_order(
        req1.symbol,
        req1.side,
        qty=req1.qty,
        time_in_force=req1.time_in_force,
        client=api,
    )
    alpaca_api.submit_order(
        req2.symbol,
        req2.side,
        qty=req2.qty,
        time_in_force=req2.time_in_force,
        client=api,
    )
    assert len(set(api.ids)) == 2
    assert hasattr(api, "client_order_ids")
    assert api.client_order_ids == api.ids


def test_shadow_mode_populates_id_lists(api, monkeypatch):
    monkeypatch.setenv("ALPACA_SHADOW", "1")
    req = make_req("MSFT")
    resp = alpaca_api.submit_order(
        req.symbol,
        req.side,
        qty=req.qty,
        time_in_force=req.time_in_force,
        client=api,
    )

    assert resp["status"] == "accepted"
    assert hasattr(api, "client_order_ids")
    assert api.ids == api.client_order_ids
    assert resp["client_order_id"] == api.ids[-1]
