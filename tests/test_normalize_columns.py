import pandas as pd
from ai_trading.data.fetch import _flatten_and_normalize_ohlcv

def test_normalize_adds_timestamp_and_volume():
    df = pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-01")], name="date"),
    )
    out = _flatten_and_normalize_ohlcv(df)
    for col in ["timestamp", "open", "high", "low", "close", "volume"]:
        assert col in out.columns


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
