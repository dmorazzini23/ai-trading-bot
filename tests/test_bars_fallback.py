import pandas as pd

from ai_trading.data.bars import _resample_minutes_to_daily


def test_resample_minute_to_daily_basic():
    idx = pd.date_range(
        "2025-08-19 13:30", periods=390 * 2, freq="1min", tz="America/New_York"
    )
    df = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000,
        },
        index=idx.tz_convert("UTC"),
    )
    out = _resample_minutes_to_daily(df)
    assert not out.empty
    assert set(["open", "high", "low", "close", "volume"]).issubset(out.columns)
