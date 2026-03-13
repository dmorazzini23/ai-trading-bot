import datetime as dt

import pandas as pd

from ai_trading.data import fetch as data_fetcher


def _frame_with_ohlcv() -> pd.DataFrame:
    index = pd.DatetimeIndex([dt.datetime(2026, 3, 11, tzinfo=dt.UTC)], name="timestamp")
    return pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [10_000],
        },
        index=index,
    )


def test_yahoo_get_bars_uses_yahoo_share_class_symbol(monkeypatch):
    requested: dict[str, list[str]] = {"symbols": []}

    def _fake_fetch(symbols, **_kwargs):
        requested["symbols"] = list(symbols)
        return {"BRK-B": _frame_with_ohlcv()}

    monkeypatch.setattr(data_fetcher, "fetch_yf_batched", _fake_fetch)

    start = dt.datetime(2026, 3, 10, tzinfo=dt.UTC)
    end = dt.datetime(2026, 3, 11, tzinfo=dt.UTC)
    frame = data_fetcher._yahoo_get_bars("BRK.B", start, end, "1d")

    assert requested["symbols"] == ["BRK-B"]
    assert not frame.empty
    assert "timestamp" in frame.columns


def test_build_daily_url_uses_yahoo_symbol_format():
    start = dt.datetime(2026, 3, 10, tzinfo=dt.UTC)
    end = dt.datetime(2026, 3, 11, tzinfo=dt.UTC)
    url = data_fetcher._build_daily_url("BRK.B", start, end)

    assert "/BRK-B?" in url
