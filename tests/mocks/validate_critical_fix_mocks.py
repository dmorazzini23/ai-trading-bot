# Extracted from scripts/validate_critical_fix.py

from unittest.mock import MagicMock

class MockOrder:
    """Simulates Alpaca order response with various filled_qty data types."""
    def __init__(self, filled_qty=None, status="filled", order_id="test-order"):
        self.filled_qty = filled_qty  # String from API (the bug cause)
        self.status = status
        self.id = order_id
        self.symbol = "TEST"

class MockContext:
    """Mock trading context."""
    def __init__(self):
        self.api = MagicMock()
        self.data_client = MagicMock()
        self.data_fetcher = MagicMock()
        self.capital_band = "small"