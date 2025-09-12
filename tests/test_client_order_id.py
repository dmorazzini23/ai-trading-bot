import types

from ai_trading import alpaca_api  # AI-AGENT-REF: canonical import


class DummyAPI:
    def __init__(self):
        self.ids: list[str] = []

    def submit_order(self, **order_data):
        self.ids.append(order_data["client_order_id"])
        return types.SimpleNamespace(id=len(self.ids), **order_data)


def make_req(symbol="AAPL"):
    return types.SimpleNamespace(symbol=symbol, qty=1, side="buy", time_in_force="day")


def test_unique_client_order_id():
    api = DummyAPI()
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
