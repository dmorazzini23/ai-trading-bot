"""Stub TradingClient for tests."""


class TradingClient:
    def __init__(self, *a, **kw):
        pass

    def get_account(self):
        return type("Account", (), {"status": "ACTIVE"})()

    def get_all_positions(self):
        return []

    def list_orders(self, *a, **kw):
        return []

    def get_orders(self, *a, **kw):
        return []

    def submit_order(self, *a, **kw):
        return {"id": "stub-order"}

    def cancel_order_by_id(self, *a, **kw):
        return True

APIError = Exception

__all__ = ["TradingClient", "APIError"]
