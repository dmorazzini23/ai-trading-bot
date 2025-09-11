import warnings
import pytest

pd = pytest.importorskip("pandas")

from ai_trading.data.fetch import _verify_minute_continuity

def _make_gap_df():
    ts = pd.date_range("2024-01-01 09:30", periods=3, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": [ts[0], ts[2]],
            "open": [1.0, 1.2],
            "high": [1.1, 1.3],
            "low": [0.9, 1.1],
            "close": [1.05, 1.25],
            "volume": [100, 150],
        }
    )

def test_warn_on_gaps():
    df = _make_gap_df()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = _verify_minute_continuity(df, "TEST")
    assert len(w) == 1
    assert out.equals(df)

def test_backfill_ffill():
    df = _make_gap_df()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        out = _verify_minute_continuity(df, "TEST", backfill="ffill")
    ts = pd.date_range("2024-01-01 09:30", periods=3, freq="1min", tz="UTC")
    assert len(out) == 3
    mid = out.loc[out["timestamp"] == ts[1]].iloc[0]
    assert mid["close"] == df["close"].iloc[0]
    assert mid["volume"] == 0

def test_backfill_interpolate():
    df = _make_gap_df()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        out = _verify_minute_continuity(df, "TEST", backfill="interpolate")
    ts = pd.date_range("2024-01-01 09:30", periods=3, freq="1min", tz="UTC")
    mid = out.loc[out["timestamp"] == ts[1]].iloc[0]
    assert mid["close"] == pytest.approx((df["close"].iloc[0] + df["close"].iloc[1]) / 2)
