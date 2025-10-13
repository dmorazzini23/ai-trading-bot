import math

import pytest

pytest.importorskip("pandas")

from ai_trading.data.fetch import _post_process, DataFetchError
from ai_trading.utils.lazy_imports import load_pandas


def test_post_process_raises_when_close_all_nan(caplog):
    pd = load_pandas()
    ts = pd.date_range("2024-01-01 09:30", periods=3, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [math.nan, math.nan, math.nan],
            "high": [math.nan, math.nan, math.nan],
            "low": [math.nan, math.nan, math.nan],
            "close": [math.nan, math.nan, math.nan],
            "volume": [1_000, 1_100, 1_050],
        }
    )

    caplog.set_level("ERROR", logger="ai_trading.data.fetch")

    with pytest.raises(DataFetchError) as excinfo:
        _post_process(df, symbol="AAPL", timeframe="1Min")

    assert getattr(excinfo.value, "fetch_reason", "") == "close_column_all_nan"
    assert any(
        getattr(record, "symbol", None) == "AAPL" and getattr(record, "timeframe", None) == "1Min"
        for record in caplog.records
    )


def test_post_process_recovers_close_from_vwap(caplog):
    pd = load_pandas()
    ts = pd.date_range("2024-01-01 09:30", periods=3, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [10.0, 10.5, 11.0],
            "high": [10.2, 10.7, 11.2],
            "low": [9.9, 10.3, 10.8],
            "close": [math.nan, math.nan, math.nan],
            "vw": [10.05, 10.55, 11.05],
            "volume": [1_000, 1_100, 1_050],
        }
    )

    caplog.set_level("INFO", logger="ai_trading.data.fetch")

    result = _post_process(df, symbol="AAPL", timeframe="1Min")

    assert result is not None
    assert "close" in result.columns
    assert result["close"].tolist() == pytest.approx([10.05, 10.55, 11.05])
    assert any(record.message == "OHLCV_CLOSE_RECOVERED" for record in caplog.records)


def test_post_process_recovers_close_from_open(caplog):
    pd = load_pandas()
    ts = pd.date_range("2024-01-01 09:30", periods=3, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [10.0, 10.5, 11.0],
            "high": [10.2, 10.7, 11.2],
            "low": [9.9, 10.3, 10.8],
            "close": [math.nan, math.nan, math.nan],
            "volume": [1_000, 1_100, 1_050],
        }
    )

    caplog.set_level("INFO", logger="ai_trading.data.fetch")

    result = _post_process(df, symbol="AAPL", timeframe="1Min")

    assert result is not None
    assert "close" in result.columns
    assert result["close"].tolist() == pytest.approx([10.0, 10.5, 11.0])
    assert any(
        record.message == "OHLCV_CLOSE_RECOVERED"
        and getattr(record, "symbol", None) == "AAPL"
        for record in caplog.records
    )


def test_post_process_handles_compact_column_names(caplog):
    pd = load_pandas()
    ts = pd.date_range("2024-01-01 09:30", periods=3, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "t": ts,
            "o": [10.0, 10.5, 11.0],
            "h": [10.2, 10.7, 11.2],
            "l": [9.9, 10.3, 10.8],
            "c": [10.1, 10.6, 11.1],
            "v": [1_000, 1_100, 1_050],
        }
    )

    caplog.set_level("INFO", logger="ai_trading.data.fetch")

    result = _post_process(df, symbol="AAPL", timeframe="1Min")

    assert result is not None
    assert "close" in result.columns
    assert result["close"].tolist() == pytest.approx([10.1, 10.6, 11.1])
    # no recovery log is necessary; ensure data parsed without raising
