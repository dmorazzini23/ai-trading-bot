import types

import pandas as pd
import pytest

from bot_engine import pre_trade_health_check


class DummyFetcher:
    def __init__(self, df):
        self.df = df
    def get_daily_df(self, ctx, sym):
        return self.df

class DummyAPI:
    def get_account(self):
        return types.SimpleNamespace()

class DummyCtx:
    def __init__(self, df):
        self.data_fetcher = DummyFetcher(df)
        self.api = DummyAPI()


def test_health_check_empty_dataframe_raises(monkeypatch):
    monkeypatch.setenv("HEALTH_MIN_ROWS", "30")
    ctx = DummyCtx(pd.DataFrame())
    with pytest.raises(RuntimeError):
        pre_trade_health_check(ctx, ["AAA"])


def test_health_check_succeeds(monkeypatch):
    monkeypatch.setenv("HEALTH_MIN_ROWS", "30")
    df = pd.DataFrame({
        "open": [1] * 30,
        "high": [1] * 30,
        "low": [1] * 30,
        "close": [1] * 30,
        "volume": [1] * 30,
    })
    ctx = DummyCtx(df)
    pre_trade_health_check(ctx, ["AAA"])

