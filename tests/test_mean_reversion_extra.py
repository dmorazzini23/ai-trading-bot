import pandas as pd

import pytest

pytest.importorskip("ai_trading.strategies.mean_reversion")
from ai_trading.strategies.mean_reversion import MeanReversionStrategy


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
    """Insufficient history skips generation."""
    df = pd.DataFrame({"close": [1, 2]})
    ctx = Ctx(df)
    strat = MeanReversionStrategy(lookback=5)
    caplog.set_level('WARNING')
    assert strat.generate(ctx) == []
    assert "insufficient" in caplog.text


def test_generate_invalid_stats(caplog):
    """Invalid rolling statistics skip generation."""
    df = pd.DataFrame({"close": [1]*10})
    ctx = Ctx(df)
    strat = MeanReversionStrategy(lookback=3)
    caplog.set_level('WARNING')
    ctx.data_fetcher.df.loc[ctx.data_fetcher.df.index[-1], "close"] = float('nan')
    assert strat.generate(ctx) == []
    assert "invalid rolling" in caplog.text


