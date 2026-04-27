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


class MockTradingClient:
    """Minimal trading client mock with failure injection."""

    def __init__(self, fail_count: int = 0, *_, **__):
        self.fail_count = fail_count
        self.call_count = 0
        self.submitted_orders: list[dict] = []

    def submit_order(self, order=None, *, order_data=None):
        self.call_count += 1
        if self.fail_count > 0:
            self.fail_count -= 1
            raise ConnectionError("mock submit failure")
        if order_data is not None:
            order = order_data
        self.submitted_orders.append(order)
        if isinstance(order, dict):
            payload = dict(order)
        else:
            payload = {
                "symbol": getattr(order, "symbol", None),
                "qty": getattr(order, "qty", getattr(order, "quantity", None)),
                "side": getattr(order, "side", None),
                "time_in_force": getattr(order, "time_in_force", None),
                "client_order_id": getattr(order, "client_order_id", None),
                "limit_price": getattr(order, "limit_price", None),
            }
        return {"id": str(self.call_count), **payload}

    def get_account(self):  # pragma: no cover - basic mock
        return {"status": "ACTIVE", "buying_power": "100000"}

    def list_positions(self):  # pragma: no cover - basic mock
        return []

__all__ = [
    "MockSignal",
    "MockModel",
    "MockContext",
    "MockContextShortSelling",
    "MockOrder",
    "MockOrderManager",
    "MockTradingClient",
]
