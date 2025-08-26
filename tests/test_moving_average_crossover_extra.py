"""Tests for moving average crossover strategy."""

import pytest

pd = pytest.importorskip("pandas")
from ai_trading.strategies.moving_average_crossover import (
    MovingAverageCrossoverStrategy,
)


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
    df = pd.DataFrame({"close": [1, 2]})
    ctx = Ctx(df)
    strat = MovingAverageCrossoverStrategy(short=3, long=5)
    caplog.set_level("WARNING")
    assert strat.generate(ctx) == []
    assert "Insufficient data" in caplog.text


def test_generate_buy_signal():
    df = pd.DataFrame({"close": [1, 2, 3, 4, 5, 6]})
    ctx = Ctx(df)
    strat = MovingAverageCrossoverStrategy(short=2, long=3)
    signals = strat.generate(ctx)
    assert signals and signals[0].side == "buy"
