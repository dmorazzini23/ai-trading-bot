import pandas as pd
import pytest

from ai_trading.data.fetch import DataFetchError, _flatten_and_normalize_ohlcv, normalize_ohlcv_columns

def test_normalize_backfills_missing_volume():
    df = pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-01")], name="date"),
    )
    out = _flatten_and_normalize_ohlcv(df, symbol="AAPL", timeframe="1Min")

    assert "volume" in out.columns
    assert (out["volume"] == 0).all()


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
    assert out["open"].tolist() == [10.0, 10.5, 10.75]
    assert out["close"].tolist() == [10.1, 10.6, 10.8]


def test_normalize_maps_dotted_aliases():
    ts = pd.date_range("2024-01-01 09:30", periods=2, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "bars.t": ts,
            "bars.open": [10.0, 10.5],
            "bars.high": [10.2, 10.7],
            "bars.low": [9.9, 10.3],
            "bars.close": [10.1, 10.6],
            "bars.volume": [1_000, 1_500],
        }
    )

    out = _flatten_and_normalize_ohlcv(df)

    assert {"timestamp", "open", "high", "low", "close", "volume"}.issubset(out.columns)
    first = out.iloc[0]
    assert pytest.approx(first["open"]) == 10.0
    assert pytest.approx(first["high"]) == 10.2
    assert pytest.approx(first["low"]) == 9.9
    assert pytest.approx(first["close"]) == 10.1
    assert pytest.approx(first["volume"]) == 1_000


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


def test_normalize_keeps_timestamp_timezone_awareness():
    ts = pd.date_range("2024-01-01 09:30", periods=3, freq="1min", tz="America/New_York")
    df = pd.DataFrame(
        {
            "open": [10.0, 10.1, 10.2],
            "high": [10.2, 10.3, 10.4],
            "low": [9.8, 9.9, 10.0],
            "close": [10.1, 10.15, 10.25],
            "volume": [1_000, 1_200, 1_100],
        },
        index=ts,
    )

    out = _flatten_and_normalize_ohlcv(df.copy())

    assert "timestamp" in out.columns
    timestamp_series = out["timestamp"]
    assert hasattr(timestamp_series, "dt")
    assert getattr(timestamp_series.dt, "tz", None) is not None
    assert str(timestamp_series.dt.tz) == "UTC"
