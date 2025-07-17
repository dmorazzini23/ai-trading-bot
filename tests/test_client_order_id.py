import types
import alpaca_api


class DummyAPI:
    def __init__(self):
        self.ids = []

    def submit_order(self, order_data=None):
        self.ids.append(order_data.client_order_id)
        return types.SimpleNamespace(id=len(self.ids), client_order_id=order_data.client_order_id)


def make_req(symbol="AAPL"):
    return types.SimpleNamespace(symbol=symbol, qty=1, side="buy", time_in_force="day")


def test_unique_client_order_id():
    api = DummyAPI()
    alpaca_api.submit_order(api, make_req())
    alpaca_api.submit_order(api, make_req())
    assert len(set(api.ids)) == 2
