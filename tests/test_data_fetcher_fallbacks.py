import datetime as dt

import pandas as pd

from ai_trading import data_fetcher as dfetch


def _fake_yf(symbol, period=None, start=None, end=None, interval="1m", **_):
    idx = pd.date_range(end=dt.datetime.now(dt.timezone.utc), periods=5, freq="1min")
    return pd.DataFrame(
        {
            "Open": [1, 2, 3, 4, 5],
            "High": [2, 3, 4, 5, 6],
            "Low": [0, 1, 2, 3, 4],
            "Close": [1.5] * 5,
            "Volume": [100] * 5,
        },
        index=idx,
    )


def test_minute_fallback_on_empty(monkeypatch):
    """Minute bars fall back to Yahoo when Alpaca yields empty."""  # AI-AGENT-REF
    monkeypatch.setattr(dfetch, "_fetch_bars", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(dfetch.yf, "download", _fake_yf)
    now = dt.datetime.now(dt.timezone.utc)
    df = dfetch.get_minute_df("ABBV", now - dt.timedelta(days=5), now)
    assert not df.empty
    assert {"open", "high", "low", "close", "volume"}.issubset({c.lower() for c in df.columns})


def test_minute_fallback_on_exception(monkeypatch):
    """Minute bars fall back to Yahoo when Alpaca errors."""  # AI-AGENT-REF
    def _boom(*_, **__):
        raise ValueError("json error")

    monkeypatch.setattr(dfetch, "_fetch_bars", _boom)
    monkeypatch.setattr(dfetch, "fh_fetcher", None)
    monkeypatch.setattr(dfetch.yf, "download", _fake_yf)
    now = dt.datetime.now(dt.timezone.utc)
    df = dfetch.get_minute_df("MSFT", now - dt.timedelta(days=1), now)
    assert not df.empty


def test_daily_fallback_on_empty(monkeypatch):
    """Daily bars fall back to Yahoo when Alpaca yields empty."""  # AI-AGENT-REF
    monkeypatch.setattr(dfetch, "get_bars", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(dfetch.yf, "download", _fake_yf)
    now = dt.datetime.now(dt.timezone.utc)
    df = dfetch.get_bars_df("SPY", now - dt.timedelta(days=30), now, timeframe="1D")
    assert not df.empty
