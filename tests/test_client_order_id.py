import types

from ai_trading import alpaca_api  # AI-AGENT-REF: canonical import


class DummyAPI:
    def __init__(self):
        self.ids = []

    def submit_order(self, **order_data):
        self.ids.append(order_data["client_order_id"])
        return types.SimpleNamespace(id=len(self.ids), **order_data)


def make_req(symbol="AAPL"):
    return types.SimpleNamespace(symbol=symbol, qty=1, side="buy", time_in_force="day")


def test_unique_client_order_id():
    api = DummyAPI()
    alpaca_api.submit_order(api, make_req())
    alpaca_api.submit_order(api, make_req())
    assert len(set(api.ids)) == 2
