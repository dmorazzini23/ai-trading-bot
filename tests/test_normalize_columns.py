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
