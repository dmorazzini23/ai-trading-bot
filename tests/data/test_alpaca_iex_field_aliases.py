from typing import Any

import pytest

pd = pytest.importorskip("pandas")

from ai_trading import alpaca_api
from ai_trading.data import fetch


class _DummyResponse:
    def __init__(self, frame):
        self.df = frame


def _alpaca_iex_raw_frame():
    idx = pd.MultiIndex.from_arrays(
        [["AAPL"], pd.to_datetime(["2024-01-02T00:00:00Z"], utc=True)],
        names=["symbol", "timestamp"],
    )
    return pd.DataFrame(
        {
            "open_price": [188.45],
            "high_price": [189.12],
            "low_price": [187.3],
            "close_price": [188.77],
            "volume": [1_234_567],
            "trade_count": [6421],
            "vwap": [188.73],
        },
        index=idx,
    )


def _bars_df_fixture():
    return _alpaca_iex_raw_frame().reset_index(drop=False)


def test_get_bars_df_alpaca_iex_columns(monkeypatch: pytest.MonkeyPatch):
    raw_frame = _alpaca_iex_raw_frame()
    expected = raw_frame.reset_index(drop=False)

    class DummyStockBarsRequest:
        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs
            for key, value in kwargs.items():
                setattr(self, key, value)

    class DummyRest:
        def __init__(self):
            self.calls: list[Any] = []

        def get_stock_bars(self, request):
            self.calls.append(request)
            return _DummyResponse(raw_frame)

    dummy_rest = DummyRest()

    monkeypatch.setattr(alpaca_api, "_get_rest", lambda bars=False: dummy_rest)
    monkeypatch.setattr(alpaca_api, "get_stock_bars_request_cls", lambda: DummyStockBarsRequest)

    df = alpaca_api.get_bars_df(
        "AAPL",
        "1Day",
        feed="alpaca_iex",
        adjustment="all",
    )

    assert dummy_rest.calls, "expected StockBarsRequest to be invoked"
    assert getattr(dummy_rest.calls[0], "feed", None) == "alpaca_iex"
    assert list(df.columns) == [
        "symbol",
        "timestamp",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "volume",
        "trade_count",
        "vwap",
    ]
    pd.testing.assert_frame_equal(df, expected)


def test_get_daily_df_normalizes_alpaca_iex_columns(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(fetch, "should_import_alpaca_sdk", lambda: True, raising=False)
    monkeypatch.setattr(fetch, "_FEED_OVERRIDE_BY_TF", {}, raising=False)

    bars_df = _bars_df_fixture()

    monkeypatch.setattr(
        alpaca_api,
        "get_bars_df",
        lambda *args, **kwargs: bars_df.copy(),
        raising=False,
    )

    df = fetch.get_daily_df("AAPL", feed="alpaca_iex")

    assert list(df.columns[:6]) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert df.index.name == "timestamp"
    assert not df.empty
    first = df.iloc[0]
    assert pytest.approx(first["open"]) == bars_df.loc[0, "open_price"]
    assert pytest.approx(first["high"]) == bars_df.loc[0, "high_price"]
    assert pytest.approx(first["low"]) == bars_df.loc[0, "low_price"]
    assert pytest.approx(first["close"]) == bars_df.loc[0, "close_price"]
    assert pytest.approx(first["volume"]) == bars_df.loc[0, "volume"]
    assert fetch._FEED_OVERRIDE_BY_TF == {}
