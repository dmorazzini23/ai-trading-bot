import datetime
import types

import pandas as pd
import pytest
from tenacity import RetryError

import data_fetcher


class DummyClient:
    def __init__(self, df):
        self.df = df
        self.calls = 0

    def get_stock_bars(self, req):
        self.calls += 1
        return types.SimpleNamespace(df=self.df)


class TF:
    Minute = "1Min"
    Hour = "1Hour"
    Day = "1Day"

    def __init__(self, *a, **k):
        pass


def make_df():
    return pd.DataFrame(
        {"open": [1.0], "high": [1.1], "low": [0.9], "close": [1.05], "volume": [10]},
        index=[pd.Timestamp("2024-01-01")],
    )


def setup_tf(monkeypatch):
    monkeypatch.setattr(data_fetcher, "TimeFrame", TF)
    monkeypatch.setattr(data_fetcher, "TimeFrameUnit", types.SimpleNamespace(Minute="m"))
    monkeypatch.setattr(data_fetcher, "StockBarsRequest", lambda **k: types.SimpleNamespace())


def test_get_historical_data(monkeypatch):
    df = make_df()
    setup_tf(monkeypatch)
    monkeypatch.setattr(data_fetcher, "_DATA_CLIENT", DummyClient(df))
    result = data_fetcher.get_historical_data("AAPL", datetime.date(2024, 1, 1), datetime.date(2024, 1, 2), "1Day")
    result = result.drop(columns=["timestamp"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(result, df.reset_index(drop=True), check_dtype=False)


def test_get_historical_data_bad_timeframe(monkeypatch):
    setup_tf(monkeypatch)
    with pytest.raises(data_fetcher.DataFetchError):
        data_fetcher.get_historical_data("AAPL", datetime.date(2024, 1, 1), datetime.date(2024, 1, 2), "10Min")


def test_get_minute_df_market_closed(monkeypatch):
    monkeypatch.setattr(data_fetcher, "is_market_open", lambda: False)
    today = datetime.date.today()
    result = data_fetcher.get_minute_df("AAPL", today, today)
    assert result.empty


def test_get_minute_df_missing_columns(monkeypatch):
    df_bad = pd.DataFrame({"price": [1]}, index=[pd.Timestamp("2024-01-01")])
    df_good = make_df()
    setup_tf(monkeypatch)
    monkeypatch.setattr(data_fetcher, "is_market_open", lambda: True)
    monkeypatch.setattr(data_fetcher, "_fetch_bars", lambda *a, **k: df_bad)
    monkeypatch.setattr(data_fetcher.fh_fetcher, "fetch", lambda *a, **k: df_good)
    data_fetcher._MINUTE_CACHE.clear()
    result = data_fetcher.get_minute_df(
        "AAPL", datetime.date(2024, 1, 1), datetime.date(2024, 1, 1)
    )
    pd.testing.assert_frame_equal(result, df_good)


def test_get_minute_df_invalid_inputs(monkeypatch):
    monkeypatch.setattr(data_fetcher, "is_market_open", lambda: True)
    with pytest.raises(ValueError):
        data_fetcher.get_minute_df("AAPL", None, datetime.date.today())
    with pytest.raises(TypeError):
        data_fetcher.get_minute_df("AAPL", 123, datetime.date.today())
