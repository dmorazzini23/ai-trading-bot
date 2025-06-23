import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for m in [
    "strategies",
    "strategies.momentum",
    "strategies.mean_reversion",
    "strategies.moving_average_crossover",
]:
    sys.modules.pop(m, None)

from strategies import (
    MeanReversionStrategy,
    MomentumStrategy,
    MovingAverageCrossoverStrategy,
    asset_class_for,
)


class DummyFetcher:
    def __init__(self, df):
        self.df = df

    def get_daily_df(self, ctx, sym):
        return self.df


class Ctx:
    def __init__(self, df):
        self.tickers = ["AAPL"]
        self.data_fetcher = DummyFetcher(df)


def test_asset_class_for():
    assert asset_class_for("EURUSD") == "forex"
    assert asset_class_for("BTCUSD") == "forex"
    assert asset_class_for("AAPL") == "equity"


def test_momentum_generate():
    df = pd.DataFrame({"close": [1, 2, 3, 4, 5]})
    ctx = Ctx(df)
    strat = MomentumStrategy(lookback=2)
    signals = strat.generate(ctx)
    assert signals and signals[0].side == "buy"


def test_mean_reversion_generate():
    df = pd.DataFrame({"close": [1, 1, 1, 1, 5]})
    ctx = Ctx(df)
    strat = MeanReversionStrategy(lookback=3, z=1.0)
    signals = strat.generate(ctx)
    assert signals and signals[0].side == "sell"
