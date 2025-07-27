import types
import alpaca_api

class MockClient:
    def __init__(self):
        self.last_payload = None
    def submit_order(self, order_data=None, *a, **k):
        self.last_payload = order_data
        return types.SimpleNamespace(id="1", status="accepted")

def test_submit_order_contract():
    api = MockClient()
    req = types.SimpleNamespace(symbol="AAPL", qty=1, side="buy", time_in_force="day")
    result = alpaca_api.submit_order(api, req)
    assert getattr(result, "id", None) == "1"
    assert getattr(api.last_payload, "symbol", None) == "AAPL"
    assert hasattr(api.last_payload, "client_order_id")
