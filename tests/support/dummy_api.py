import types


class DummyAPI:
    """Minimal Alpaca-like API for safe_submit_order tests."""

    def __init__(self):
        # Simple account state; tests stub these methods directly when needed
        self.get_account = lambda: types.SimpleNamespace(buying_power="1000")
        self.list_positions = lambda: []
        self._order = None

    def submit_order(self, symbol: str, **_kwargs):
        """Record symbol and return a pending_new order object."""
        self._order = types.SimpleNamespace(
            id=1,
            status="pending_new",
            filled_qty=0,
            symbol=symbol,
        )
        return self._order

    def get_order(self, order_id):
        """Return order, transitioning it to filled after first poll."""
        if self._order and getattr(self._order, "id", None) == order_id:
            # Simulate broker updating status on subsequent poll
            self._order.status = "filled"
            return self._order
        return types.SimpleNamespace(id=order_id, status="filled", filled_qty=0, symbol=None)
