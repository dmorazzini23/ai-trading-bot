from datetime import UTC, datetime, timedelta

import pandas as pd

from ai_trading.core.data_contract import normalize_bars, validate_bars


def _sample_df(start: datetime, periods: int = 3):
    idx = pd.date_range(start=start, periods=periods, freq="1min", tz=UTC)
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000, 1100, 1200],
        },
        index=idx,
    )


def test_validate_bars_ok():
    df = _sample_df(datetime.now(UTC) - timedelta(minutes=2))
    result = validate_bars(df, "1Min", freshness_seconds=300, rth_only=False)
    assert result.ok


def test_validate_bars_duplicate():
    df = _sample_df(datetime.now(UTC) - timedelta(minutes=2))
    df.index = [df.index[0], df.index[0], df.index[2]]
    result = validate_bars(df, "1Min", freshness_seconds=300, rth_only=False)
    assert result.ok is False
    assert result.reason == "DUPLICATE_BARS"


def test_validate_bars_stale():
    df = _sample_df(datetime.now(UTC) - timedelta(days=5))
    result = validate_bars(df, "1Day", freshness_seconds=60, rth_only=False)
    assert result.ok is False
    assert result.reason == "STALE_BAR"


def test_normalize_bars_lowercase():
    df = _sample_df(datetime.now(UTC) - timedelta(minutes=2))
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    normalized = normalize_bars(df, "1Min", tz=UTC, rth_only=False)
    assert set(["open", "high", "low", "close", "volume"]).issubset(normalized.columns)
