import pandas as pd
from ai_trading.strategies import momentum
from ai_trading.strategies.momentum import MomentumStrategy


class DummyFetcher:
    def __init__(self, df):
        self.df = df
    def get_daily_df(self, ctx, sym):
        return self.df

class Ctx:
    def __init__(self, df):
        self.tickers = ["A"]
        self.data_fetcher = DummyFetcher(df)


def test_generate_insufficient_data(caplog):
    df = pd.DataFrame({"close": [1]})
    ctx = Ctx(df)
    strat = MomentumStrategy(lookback=2)
    caplog.set_level('WARNING')
    assert strat.generate(ctx) == []
    assert "Insufficient data" in caplog.text


def test_generate_ret_nan(monkeypatch):
    df = pd.DataFrame({"close": [1,2,3,4]})
    ctx = Ctx(df)
    strat = MomentumStrategy(lookback=3)
    ctx.data_fetcher.df.loc[len(df)-1, "close"] = float('nan')
    monkeypatch.setattr(momentum.pd, "isna", lambda v: True)
    assert strat.generate(ctx) == []


def test_threshold_skip():
    df = pd.DataFrame({"close": [1, 1.02, 1.03]})
    ctx = Ctx(df)
    strat = MomentumStrategy(lookback=1, threshold=0.05)
    assert strat.generate(ctx) == []
