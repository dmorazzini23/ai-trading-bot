"""Central test mocks."""

class MockSignal:
    def __init__(self, symbol: str = "AAPL", side: str = "buy", confidence: float = 0.0):
        self.symbol = symbol
        self.side = side
        self.action = side
        self.confidence = confidence

class MockModel:
    def predict(self, *_args, **_kwargs):
        return []

class MockContext:
    def __init__(self):
        self.api = None
        self.data_client = None
        self.data_fetcher = None
        self.allow_short_selling = False

class MockContextShortSelling(MockContext):
    def __init__(self):
        super().__init__()
        self.allow_short_selling = True

class MockOrder:
    def __init__(self, filled_qty=None, status="filled", order_id="test_order"):
        self.filled_qty = filled_qty
        self.status = status
        self.id = order_id
        self.symbol = "TEST"

class MockOrderManager:
    def submit_order(self, order):
        return True

__all__ = [
    "MockSignal",
    "MockModel",
    "MockContext",
    "MockContextShortSelling",
    "MockOrder",
    "MockOrderManager",
]
