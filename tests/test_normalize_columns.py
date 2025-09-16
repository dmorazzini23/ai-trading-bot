import pandas as pd
import pytest

from ai_trading.data.fetch import DataFetchError, _flatten_and_normalize_ohlcv, normalize_ohlcv_columns

def test_normalize_requires_volume_column():
    df = pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-01")], name="date"),
    )
    with pytest.raises(DataFetchError) as excinfo:
        _flatten_and_normalize_ohlcv(df, symbol="AAPL", timeframe="1Min")

    assert getattr(excinfo.value, "fetch_reason", "") in {
        "ohlcv_columns_missing",
        "close_column_missing",
    }


def test_normalize_removes_timestamp_index_conflict():
    ts = pd.date_range("2024-01-01 09:30", periods=2, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [1.0, 1.1],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
            "close": [1.05, 1.15],
            "volume": [100, 120],
        },
        index=pd.DatetimeIndex(ts, name="timestamp"),
    )
    out = _flatten_and_normalize_ohlcv(df)
    assert "timestamp" in out.columns
    assert all(name != "timestamp" for name in (out.index.names or []))
    # Should not raise after normalize when sorting by timestamp column.
    out.sort_values("timestamp")


def test_normalize_maps_provider_aliases():
    ts = pd.date_range("2024-01-01 09:30", periods=3, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "t": ts,
            "o": [10.0, 10.5, 10.75],
            "h": [10.2, 10.7, 10.9],
            "l": [9.9, 10.3, 10.6],
            "c": [10.1, 10.6, 10.8],
            "v": [1_000, 2_000, 1_500],
        }
    )
    out = _flatten_and_normalize_ohlcv(df)
    assert {"timestamp", "open", "high", "low", "close", "volume"}.issubset(out.columns)
    assert not {"t", "o", "h", "l", "c", "v"} & set(out.columns)
    pd.testing.assert_series_equal(out["open"], pd.Series([10.0, 10.5, 10.75]), check_names=False)
    pd.testing.assert_series_equal(out["close"], pd.Series([10.1, 10.6, 10.8]), check_names=False)


def test_normalize_rejects_string_nan_close_values():
    ts = pd.date_range("2024-01-01 09:30", periods=2, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": ["1.0", "1.1"],
            "high": ["1.1", "1.2"],
            "low": ["0.9", "1.0"],
            "close": ["nan", "nan"],
            "volume": ["100", "120"],
        },
        index=ts,
    )

    with pytest.raises(DataFetchError) as excinfo:
        _flatten_and_normalize_ohlcv(df.copy())

    assert getattr(excinfo.value, "fetch_reason", "") == "close_column_all_nan"


def test_normalize_alias_helper_exposed():
    df = pd.DataFrame({"O": [1], "H": [2], "L": [0], "C": [1.5], "V": [100], "T": ["2024-01-01"]})
    out = normalize_ohlcv_columns(df.copy())
    assert {"open", "high", "low", "close", "volume", "timestamp"}.issubset(out.columns)
