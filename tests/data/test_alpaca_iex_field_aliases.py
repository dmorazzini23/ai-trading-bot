import logging
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
            "open": [188.45],
            "high": [189.12],
            "low": [187.3],
            "close": [188.77],
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
        "open",
        "high",
        "low",
        "close",
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
    assert pytest.approx(first["open"]) == bars_df.loc[0, "open"]
    assert pytest.approx(first["high"]) == bars_df.loc[0, "high"]
    assert pytest.approx(first["low"]) == bars_df.loc[0, "low"]
    assert pytest.approx(first["close"]) == bars_df.loc[0, "close"]
    assert pytest.approx(first["volume"]) == bars_df.loc[0, "volume"]
    assert fetch._FEED_OVERRIDE_BY_TF == {}


def test_ensure_ohlcv_schema_handles_camelcase_payload(caplog: pytest.LogCaptureFixture):
    payload = [
        {
            "t": "2024-01-02T09:30:00Z",
            "openPrice": 188.45,
            "highPrice": 189.12,
            "lowPrice": 187.3,
            "closePrice": 188.77,
            "totalVolume": 1_234_567,
            "tradeCount": 6421,
            "vwap": 188.73,
        }
    ]
    frame = pd.DataFrame(payload)
    fetch._attach_payload_metadata(
        frame,
        payload=payload,
        provider="alpaca",
        feed="iex",
        timeframe="1Min",
        symbol="AAPL",
    )

    with caplog.at_level(logging.ERROR):
        normalized = fetch.ensure_ohlcv_schema(frame, source="alpaca_iex", frequency="1Min")

    assert list(normalized.columns[:6]) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert not any(record.message == "OHLCV_COLUMNS_MISSING" for record in caplog.records)
    first = normalized.iloc[0]
    assert pytest.approx(first["open"]) == payload[0]["openPrice"]
    assert pytest.approx(first["high"]) == payload[0]["highPrice"]
    assert pytest.approx(first["low"]) == payload[0]["lowPrice"]
    assert pytest.approx(first["close"]) == payload[0]["closePrice"]
    assert pytest.approx(first["volume"]) == payload[0]["totalVolume"]


def test_ensure_ohlcv_schema_handles_iex_prefixed_payload(caplog: pytest.LogCaptureFixture):
    payload = [
        {
            "t": "2024-01-02T09:30:00Z",
            "iexOpen": 188.45,
            "iexHigh": 189.12,
            "iexLow": 187.3,
            "iexClose": 188.77,
            "iexVolume": 1_234_567,
            "tradeCount": 6421,
            "vwap": 188.73,
        }
    ]
    frame = pd.DataFrame(payload)
    fetch._attach_payload_metadata(
        frame,
        payload=payload,
        provider="alpaca",
        feed="iex",
        timeframe="1Min",
        symbol="AAPL",
    )

    with caplog.at_level(logging.ERROR):
        normalized = fetch.ensure_ohlcv_schema(frame, source="alpaca_iex", frequency="1Min")

    assert list(normalized.columns[:6]) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert not any(record.message == "OHLCV_COLUMNS_MISSING" for record in caplog.records)
    first = normalized.iloc[0]
    assert pytest.approx(first["open"]) == payload[0]["iexOpen"]
    assert pytest.approx(first["high"]) == payload[0]["iexHigh"]
    assert pytest.approx(first["low"]) == payload[0]["iexLow"]
    assert pytest.approx(first["close"]) == payload[0]["iexClose"]
    assert pytest.approx(first["volume"]) == payload[0]["iexVolume"]


def test_ensure_ohlcv_schema_handles_latest_price_payload(caplog: pytest.LogCaptureFixture):
    payload = [
        {
            "t": "2024-01-02T09:30:00Z",
            "sessionOpen": 188.45,
            "sessionHigh": 189.12,
            "sessionLow": 187.3,
            "latestPrice": 188.77,
            "sessionVolume": 1_234_567,
            "tradeCount": 6421,
        }
    ]
    frame = pd.DataFrame(payload)
    fetch._attach_payload_metadata(
        frame,
        payload=payload,
        provider="alpaca",
        feed="iex",
        timeframe="1Min",
        symbol="AAPL",
    )

    with caplog.at_level(logging.ERROR):
        normalized = fetch.ensure_ohlcv_schema(frame, source="alpaca_iex", frequency="1Min")

    assert list(normalized.columns[:6]) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert not any(record.message == "OHLCV_COLUMNS_MISSING" for record in caplog.records)
    first = normalized.iloc[0]
    assert pytest.approx(first["open"]) == payload[0]["sessionOpen"]
    assert pytest.approx(first["high"]) == payload[0]["sessionHigh"]
    assert pytest.approx(first["low"]) == payload[0]["sessionLow"]
    assert pytest.approx(first["close"]) == payload[0]["latestPrice"]
    assert pytest.approx(first["volume"]) == payload[0]["sessionVolume"]


def test_ensure_ohlcv_schema_handles_dotted_payload(caplog: pytest.LogCaptureFixture):
    payload = [
        {
            "t": "2024-01-02T09:30:00Z",
            "bars.open": 188.45,
            "bars.high": 189.12,
            "bars.low": 187.3,
            "bars.close": 188.77,
            "bars.volume": 1_234_567,
        }
    ]
    frame = pd.DataFrame(payload)
    fetch._attach_payload_metadata(
        frame,
        payload=payload,
        provider="alpaca",
        feed="iex",
        timeframe="1Min",
        symbol="AAPL",
    )

    with caplog.at_level(logging.ERROR):
        normalized = fetch.ensure_ohlcv_schema(frame, source="alpaca_iex", frequency="1Min")

    assert list(normalized.columns[:6]) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert not any(record.message == "OHLCV_COLUMNS_MISSING" for record in caplog.records)
    first = normalized.iloc[0]
    assert pytest.approx(first["open"]) == payload[0]["bars.open"]
    assert pytest.approx(first["high"]) == payload[0]["bars.high"]
    assert pytest.approx(first["low"]) == payload[0]["bars.low"]
    assert pytest.approx(first["close"]) == payload[0]["bars.close"]
    assert pytest.approx(first["volume"]) == payload[0]["bars.volume"]


def test_ensure_ohlcv_schema_handles_compact_payload():
    payload = [
        {"t": "2024-01-02T09:30:00Z", "o": 188.45, "h": 189.12, "l": 187.3, "c": 188.77, "v": 1_234_567}
    ]
    frame = pd.DataFrame(payload)
    fetch._attach_payload_metadata(
        frame,
        payload=payload,
        provider="alpaca",
        feed="iex",
        timeframe="1Min",
        symbol="AAPL",
    )

    normalized = fetch.ensure_ohlcv_schema(frame, source="alpaca_iex", frequency="1Min")

    assert list(normalized.columns[:6]) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    first = normalized.iloc[0]
    assert pytest.approx(first["open"]) == payload[0]["o"]
    assert pytest.approx(first["high"]) == payload[0]["h"]
    assert pytest.approx(first["low"]) == payload[0]["l"]
    assert pytest.approx(first["close"]) == payload[0]["c"]
    assert pytest.approx(first["volume"]) == payload[0]["v"]


def test_ensure_ohlcv_schema_handles_two_letter_payload(caplog: pytest.LogCaptureFixture):
    payload = [
        {
            "t": "2024-01-02T09:30:00Z",
            "op": 188.45,
            "hi": 189.12,
            "lo": 187.3,
            "cl": 188.77,
            "vol": 1_234_567,
            "n": 6_421,
            "vw": 188.73,
        }
    ]
    frame = pd.DataFrame(payload)
    fetch._attach_payload_metadata(
        frame,
        payload=payload,
        provider="alpaca",
        feed="iex",
        timeframe="1Min",
        symbol="AAPL",
    )

    with caplog.at_level(logging.ERROR):
        normalized = fetch.ensure_ohlcv_schema(frame, source="alpaca_iex", frequency="1Min")

    assert list(normalized.columns[:6]) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert not any(record.message == "OHLCV_COLUMNS_MISSING" for record in caplog.records)
    first = normalized.iloc[0]
    assert pytest.approx(first["open"]) == payload[0]["op"]
    assert pytest.approx(first["high"]) == payload[0]["hi"]
    assert pytest.approx(first["low"]) == payload[0]["lo"]
    assert pytest.approx(first["close"]) == payload[0]["cl"]
    assert pytest.approx(first["volume"]) == payload[0]["vol"]


def test_ensure_ohlcv_schema_handles_first_highest_lowest_payload(
    caplog: pytest.LogCaptureFixture,
):
    payload = [
        {
            "t": "2024-01-02T09:30:00Z",
            "firstPrice": 188.45,
            "highestPrice": 189.12,
            "lowestPrice": 187.3,
            "lastPrice": 188.77,
            "accumulatedVolume": 1_234_567,
        }
    ]
    frame = pd.DataFrame(payload)
    fetch._attach_payload_metadata(
        frame,
        payload=payload,
        provider="alpaca",
        feed="iex",
        timeframe="1Min",
        symbol="AAPL",
    )

    with caplog.at_level(logging.ERROR):
        normalized = fetch.ensure_ohlcv_schema(
            frame, source="alpaca_iex", frequency="1Min"
        )

    assert list(normalized.columns[:6]) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert not any(record.message == "OHLCV_COLUMNS_MISSING" for record in caplog.records)
    first = normalized.iloc[0]
    assert pytest.approx(first["open"]) == payload[0]["firstPrice"]
    assert pytest.approx(first["high"]) == payload[0]["highestPrice"]
    assert pytest.approx(first["low"]) == payload[0]["lowestPrice"]
    assert pytest.approx(first["close"]) == payload[0]["lastPrice"]
    assert pytest.approx(first["volume"]) == payload[0]["accumulatedVolume"]


def test_ensure_ohlcv_schema_logs_payload_columns(caplog: pytest.LogCaptureFixture):
    payload = [
        {
            "t": "2024-01-02T09:30:00Z",
            "volume": 1_000,
        }
    ]
    frame = pd.DataFrame(payload)
    fetch._attach_payload_metadata(
        frame,
        payload=payload,
        provider="alpaca",
        feed="iex",
        timeframe="1Min",
        symbol="AAPL",
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(fetch.MissingOHLCVColumnsError) as excinfo:
            fetch.ensure_ohlcv_schema(frame, source="alpaca_iex", frequency="1Min")

    records = [record for record in caplog.records if record.message == "OHLCV_COLUMNS_MISSING"]
    assert records, "expected OHLCV_COLUMNS_MISSING to be logged"
    logged = records[-1]
    assert getattr(logged, "raw_payload_columns", None) == ["t", "volume"]
    assert getattr(logged, "raw_payload_feed", None) == "iex"
    assert getattr(logged, "raw_payload_timeframe", None) == "1Min"
    assert getattr(logged, "raw_payload_symbol", None) == "AAPL"
    err = excinfo.value
    assert getattr(err, "raw_payload_columns", None) == ("t", "volume")
    assert getattr(err, "raw_payload_feed", None) == "iex"
    assert getattr(err, "raw_payload_symbol", None) == "AAPL"
