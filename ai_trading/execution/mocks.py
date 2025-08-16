from types import SimpleNamespace


class MockContext:
    """Lightweight execution context used in tests."""  # AI-AGENT-REF: test context

    def __init__(self):
        self.positions: dict = {}
        self.orders: list = []


class MockTradingClient:
    """Test double for Alpaca TradingClient."""  # AI-AGENT-REF: minimal trading mock

    def __init__(self, *a, **k):
        self._orders: list = []
        self._id = 0

    def submit_order(self, **kw):
        self._id += 1
        ns = SimpleNamespace(id=str(self._id), status="accepted", **kw)
        self._orders.append(ns)
        return ns

    def get_account(self):
        return SimpleNamespace(status="ACTIVE", buying_power="100000")

    def list_positions(self):
        return []

    def get_order_by_id(self, order_id):
        return SimpleNamespace(id=str(order_id), status="filled")


__all__ = ["MockContext", "MockTradingClient"]
