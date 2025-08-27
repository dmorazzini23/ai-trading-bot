"""Backward compatibility stub mapping to alpaca-py style classes."""
from tests.vendor_stubs.alpaca.trading.client import TradingClient, APIError

# Legacy name used by code; alias TradingClient as REST for compatibility
REST = TradingClient

__all__ = ["TradingClient", "REST", "APIError"]
