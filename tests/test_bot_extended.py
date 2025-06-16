import types
import pandas as pd

# Reuse stubs from existing bot tests to avoid heavy imports
from tests import test_bot as base
bot = base.bot

def test_compute_time_range():
    start, end = bot.compute_time_range(5)
    assert (end - start).total_seconds() == 300


def test_get_latest_close_edge_cases():
    assert bot.get_latest_close(pd.DataFrame()) == 0.0
    assert bot.get_latest_close(None) == 0.0
    df = pd.DataFrame({'close':[1.5]}, index=[pd.Timestamp('2024-01-01')])
    assert bot.get_latest_close(df) == 1.5


class DummyDFetch:
    def __init__(self, df):
        self.df = df
    def get_minute_df(self, ctx, symbol):
        return self.df

class DummyCtx:
    def __init__(self, df=pd.DataFrame()):
        self.data_fetcher = DummyDFetch(df)

def test_fetch_minute_df_safe_market_closed(monkeypatch):
    monkeypatch.setattr(bot, 'market_is_open', lambda now=None: False)
    ctx = DummyCtx()
    result = bot.fetch_minute_df_safe(ctx, 'AAPL')
    assert result.empty


def test_fetch_minute_df_safe_open(monkeypatch):
    monkeypatch.setattr(bot, 'market_is_open', lambda now=None: True)
    df = pd.DataFrame({'close':[1]}, index=[pd.Timestamp('2024-01-01')])
    ctx = DummyCtx(df)
    result = bot.fetch_minute_df_safe(ctx, 'AAPL')
    pd.testing.assert_frame_equal(result, df)


def test_cancel_all_open_orders(monkeypatch):
    class API:
        def get_orders(self, req):
            return [types.SimpleNamespace(id=1, status='open')]
        def cancel_order_by_id(self, oid):
            self.cancelled = oid
    ctx = types.SimpleNamespace(api=API())
    bot.cancel_all_open_orders(ctx)
    assert getattr(ctx.api, 'cancelled', None) == 1


def test_reconcile_positions(monkeypatch):
    positions = [types.SimpleNamespace(symbol='AAPL', qty=0)]
    api = types.SimpleNamespace(get_all_positions=lambda: positions)
    ctx = types.SimpleNamespace(api=api, stop_targets={'AAPL':1}, take_profit_targets={'AAPL':1})
    bot.reconcile_positions(ctx)
    assert not ctx.stop_targets and not ctx.take_profit_targets

