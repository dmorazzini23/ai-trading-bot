from ai_trading.broker.alpaca import AlpacaBroker


from ai_trading.broker.alpaca import APIError


class FakeClient:
    def __init__(self, positions):
        self._positions = positions

    def list_positions(self):
        return self._positions

    def get_position(self, symbol):
        for p in self._positions:
            if p.symbol == symbol:
                return p
        raise APIError({"code": 404, "message": "not found"}, None)


class Obj:
    def __init__(self, symbol, qty, avg_entry_price):
        self.symbol = symbol
        self.qty = qty
        self.avg_entry_price = avg_entry_price


def test_list_open_positions_returns_objects():
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._api = FakeClient([Obj("AAPL", "10", "150.0")])
    broker._is_new = False
    out = broker.list_open_positions()
    assert out and hasattr(out[0], "symbol") and hasattr(out[0], "qty")


def test_get_open_position_uses_sdk_then_falls_back():
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._api = FakeClient([Obj("MSFT", "5", "300.0")])
    broker._is_new = False
    pos = broker.get_open_position("MSFT")
    assert pos and pos.symbol == "MSFT" and int(pos.qty) == 5
    none = broker.get_open_position("NVDA")
    assert none is None
