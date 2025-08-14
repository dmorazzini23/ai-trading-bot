import types
import pandas as pd
import datetime
from ai_trading.core import bot_engine

class DummyFetcher:
    def get_daily_df(self, ctx, symbol):
        return pd.DataFrame({"close": [100]})

class DummyAPI:
    def __init__(self):
        self.positions = {}
        self.orders = []
    def get_account(self):
        return types.SimpleNamespace(cash=1000.0, equity=1000.0, buying_power=1000.0)
    def get_all_positions(self):
        return [types.SimpleNamespace(symbol=s, qty=q) for s, q in self.positions.items()]


def test_partial_initial_rebalance_fill(monkeypatch):
    ctx = types.SimpleNamespace(
        api=DummyAPI(),
        data_fetcher=DummyFetcher(),
        rebalance_ids={},
        rebalance_attempts={},
        rebalance_buys={},
    )

    class FakeDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.datetime(2025, 7, 26, 0, 16, tzinfo=datetime.UTC)

    monkeypatch.setattr(bot_engine, "datetime", FakeDateTime)

    def fake_submit(ctx_, symbol, qty, side):
        ctx_.api.orders.append((symbol, qty, side))
        ctx_.api.positions[symbol] = ctx_.api.positions.get(symbol, 0) + qty // 2
        return object()

    monkeypatch.setattr(bot_engine, "submit_order", fake_submit)

    bot_engine.initial_rebalance(ctx, ["AAPL"])
    assert ctx.api.positions["AAPL"] == 5

    bot_engine.initial_rebalance(ctx, ["AAPL"])
    assert all(o[2] == "buy" for o in ctx.api.orders)
    assert ctx.api.positions["AAPL"] == 10
