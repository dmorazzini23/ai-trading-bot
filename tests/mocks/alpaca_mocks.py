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

    def submit_order(self, order_data=None, **order):
        if order_data is not None:
            order = {"order_data": order_data, **order}
        self.orders.append(order)
        return {"status": "accepted", "id": f"mock-{len(self.orders)}"}


