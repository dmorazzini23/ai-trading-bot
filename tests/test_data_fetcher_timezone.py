import sys
import types
from datetime import UTC, datetime

import pytest

pd = pytest.importorskip("pandas")
# AI-AGENT-REF: regression test for Yahoo timezone normalization


def _install_fake_yf(monkeypatch):
    """Install a minimal fake yfinance module into sys.modules."""
    fake = types.SimpleNamespace()

    def _download(symbol, start=None, end=None, interval=None, progress=None):
        assert start is None or start.tzinfo is UTC
        assert end is None or end.tzinfo is UTC
        idx_name = "Date" if interval in (None, "1d") else "Datetime"
        idx = pd.date_range("2025-08-01", periods=2, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "Open": [1.0, 2.0],
                "High": [1.5, 2.5],
                "Low": [0.5, 1.5],
                "Close": [1.2, 2.2],
                "Adj Close": [1.2, 2.2],
                "Volume": [100, 200],
            },
            index=idx,
        )
        df.index.name = idx_name
        return df

    fake.download = _download
    monkeypatch.setitem(sys.modules, "yfinance", fake)


def test_yahoo_get_bars_accepts_various_datetime_types(monkeypatch):
    from ai_trading.data_fetcher import _yahoo_get_bars

    _install_fake_yf(monkeypatch)

    df1 = _yahoo_get_bars("SPY", datetime(2025, 8, 1), datetime(2025, 8, 10), "1Day")
    assert not df1.empty and df1["timestamp"].dt.tz is not None

    df2 = _yahoo_get_bars(
        "SPY", datetime(2025, 8, 1, tzinfo=UTC), datetime(2025, 8, 10, tzinfo=UTC), "1Day"
    )
    assert not df2.empty and df2["timestamp"].dt.tz is not None

    df3 = _yahoo_get_bars("SPY", pd.Timestamp("2025-08-01"), pd.Timestamp("2025-08-10"), "1Day")
    assert not df3.empty and df3["timestamp"].dt.tz is not None

    df4 = _yahoo_get_bars(
        "SPY",
        pd.Timestamp("2025-08-01", tz="UTC"),
        pd.Timestamp("2025-08-10", tz="UTC"),
        "1Day",
    )
    assert not df4.empty and df4["timestamp"].dt.tz is not None
