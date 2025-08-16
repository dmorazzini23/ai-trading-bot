from types import SimpleNamespace

class MockTradingClient:
    """Test double for Alpaca TradingClient."""  # AI-AGENT-REF: minimal trading mock
    def __init__(self, *a, **k):
        self._orders = []

    def submit_order(self, symbol, qty, side, time_in_force, **kw):
        oid = str(len(self._orders) + 1)
        self._orders.append(oid)
        return SimpleNamespace(id=oid, status="accepted", symbol=symbol, qty=qty, side=side)

    def get_account(self):
        return SimpleNamespace(status="ACTIVE", buying_power="100000")

    def list_positions(self):
        return []

    def get_order_by_id(self, order_id):
        return SimpleNamespace(id=str(order_id), status="filled")
