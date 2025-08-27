"""Stub TradingClient for tests."""

class TradingClient:
    def __init__(self, *a, **kw):
        pass

    def get_all_positions(self):
        return []

APIError = Exception

__all__ = ["TradingClient", "APIError"]
