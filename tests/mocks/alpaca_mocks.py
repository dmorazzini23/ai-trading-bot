class MockOrderSide:
    BUY = "buy"
    SELL = "sell"


class MockTimeInForce:
    DAY = "day"
    GTC = "gtc"


class MockOrderStatus:
    NEW = "new"
    FILLED = "filled"
    CANCELED = "canceled"


class MockQueryOrderStatus:
    OPEN = "open"
    CLOSED = "closed"


class MockTradingClient:
    def __init__(self, *args, **kwargs):
        self.orders = []

    def submit_order(self, **order):
        self.orders.append(order)
        return {"status": "accepted", "id": f"mock-{len(self.orders)}"}


import types


class MockClient:
    """Replaces ai_trading/alpaca_contract.py:MockClient for tests."""  # AI-AGENT-REF: normalize id

    def __init__(self):
        self.last_payload = None

    def submit_order(self, **order):
        self.last_payload = types.SimpleNamespace(**order)
        return {"status": "accepted", "id": "1"}
