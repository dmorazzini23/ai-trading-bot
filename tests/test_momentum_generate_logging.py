import pytest
pd = pytest.importorskip("pandas")
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


def test_generate_logs_insufficient_data(caplog):
    df = pd.DataFrame({"close": [1]})
    ctx = Ctx(df)
    strat = MomentumStrategy(lookback=2)
    caplog.set_level("WARNING")
    assert strat.generate(ctx) == []
    assert "Insufficient data" in caplog.text
