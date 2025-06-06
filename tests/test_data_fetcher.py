import sys
import types
from pathlib import Path
import pandas as pd
import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
mods = [
    "alpaca",
    "alpaca.data.historical",
    "alpaca.data.requests",
    "alpaca.data.timeframe",
    "alpaca_trade_api.rest",
    "finnhub",
]
for m in mods:
    sys.modules.setdefault(m, types.ModuleType(m))
sys.modules.setdefault("alpaca_trade_api", types.ModuleType("alpaca_trade_api"))


class _FakeREST:
    def __init__(self, *a, **k):
        pass

    def get_bars(self, *a, **k):
        return types.SimpleNamespace(df=pd.DataFrame())


sys.modules["alpaca_trade_api"].REST = _FakeREST
sys.modules["alpaca_trade_api"].APIError = Exception
sys.modules["alpaca_trade_api.rest"].REST = _FakeREST
sys.modules["alpaca_trade_api.rest"].APIError = Exception
sys.modules["alpaca_trade_api.rest"].TimeFrame = object


class _DummyHist:
    def __init__(self, *a, **k):
        pass

    def get_stock_bars(self, *a, **k):
        import pandas as pd

        return types.SimpleNamespace(df=pd.DataFrame())


sys.modules["alpaca.data.historical"].StockHistoricalDataClient = _DummyHist
sys.modules["alpaca.data.requests"].StockBarsRequest = object
sys.modules["alpaca.data.requests"].StockLatestQuoteRequest = object
sys.modules["alpaca.data.timeframe"].TimeFrame = object
sys.modules["alpaca.data.timeframe"].TimeFrameUnit = object


class _DummyFinnhub:
    def __init__(self, *a, **k):
        pass


sys.modules["finnhub"].Client = _DummyFinnhub

import data_fetcher


class FakeBars:
    def __init__(self, df: pd.DataFrame):
        self.df = df


def test_get_minute_df(monkeypatch):
    df = pd.DataFrame(
        {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100]},
        index=[pd.Timestamp("2023-01-01T09:30")],
    )

    def fake_get_bars(*args, **kwargs):
        return FakeBars(df)

    monkeypatch.setattr(data_fetcher.client, "get_bars", fake_get_bars)
    result = data_fetcher.get_minute_df(
        "AAPL", datetime.date(2023, 1, 1), datetime.date(2023, 1, 2)
    )
    pd.testing.assert_frame_equal(result, df)
