import sys
import types

rebalancer_stub = types.ModuleType("ai_trading.rebalancer")
rebalancer_stub.maybe_rebalance = lambda *a, **k: None
sys.modules.setdefault("ai_trading.rebalancer", rebalancer_stub)

from ai_trading.core import bot_engine


class DummyPosition:
    def __init__(self, symbol, qty):
        self.symbol = symbol
        self.qty = qty


class DummyAPI:
    def __init__(self):
        self.called = False

    def get_all_positions(self):
        self.called = True
        return [DummyPosition("AAPL", "3")]

    def list_positions(self):
        raise AssertionError("list_positions should not be used")


def test_audit_positions_uses_get_all_positions(monkeypatch):
    monkeypatch.setattr(bot_engine, "_parse_local_positions", lambda: {"AAPL": 1})

    class _TIF:
        DAY = "day"

    class _Side:
        BUY = "buy"
        SELL = "sell"

    class _Req:
        def __init__(self, symbol, qty, side, time_in_force):
            self.symbol = symbol
            self.qty = qty
            self.side = side
            self.time_in_force = time_in_force

    monkeypatch.setattr(bot_engine, "TimeInForce", _TIF)
    monkeypatch.setattr(bot_engine, "OrderSide", _Side)
    monkeypatch.setattr(bot_engine, "MarketOrderRequest", _Req)

    orders = []

    def fake_submit(api, req):
        orders.append(req)

    monkeypatch.setattr(bot_engine, "safe_submit_order", fake_submit)

    api = DummyAPI()
    ctx = types.SimpleNamespace(api=api)
    bot_engine.runtime = ctx

    bot_engine.audit_positions(ctx)

    assert api.called
    assert orders
    order = orders[0]
    assert order.symbol == "AAPL"
    assert int(order.qty) == 2
    assert order.side == bot_engine.OrderSide.SELL
