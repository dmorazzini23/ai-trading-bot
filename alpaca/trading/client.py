"""Minimal TradingClient stub used in tests."""

class TradingClient:  # pragma: no cover - trivial
    def __init__(self, *args, **kwargs):
        self._orders = []

    def list_orders(self, *a, **k):
        return []

    def list_positions(self, *a, **k):
        return []

    def submit_order(self, *a, **k):  # noqa: D401
        """Return simple order namespace."""
        from types import SimpleNamespace

        order = SimpleNamespace(id="0", status="accepted", **k)
        self._orders.append(order)
        return order

    def get_order(self, order_id):
        for o in self._orders:
            if str(getattr(o, "id", "")) == str(order_id):
                return o
        from types import SimpleNamespace

        return SimpleNamespace(id=order_id, status="filled")
