from __future__ import annotations

from datetime import UTC, datetime, timedelta

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from ai_trading.data import fetch as data_fetcher


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    monkeypatch.delenv("MINUTE_SOURCE", raising=False)
    monkeypatch.delenv("DAILY_SOURCE", raising=False)
    monkeypatch.delenv("ALPACA_DATA_FEED", raising=False)
    monkeypatch.delenv("DATA_FEED_INTRADAY", raising=False)
    yield


def _stub_frame() -> pd.DataFrame:
    ts_start = datetime.now(UTC) - timedelta(minutes=5)
    ts_end = ts_start + timedelta(minutes=1)
    return pd.DataFrame(
        {
            "timestamp": [ts_start, ts_end],
            "open": [1.0, 1.1],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
            "close": [1.05, 1.15],
            "volume": [100, 120],
        }
    )


def test_minute_source_override_yahoo(monkeypatch):
    monkeypatch.setenv("MINUTE_SOURCE", "yahoo")
    fetch_calls = {"count": 0}

    def _count_fetch(*args, **kwargs):
        fetch_calls["count"] += 1
        return None

    monkeypatch.setattr(data_fetcher, "_fetch_bars", _count_fetch)

    stub_df = _stub_frame()

    def fake_yahoo(symbol, start, end, interval):
        return stub_df.copy()

    monkeypatch.setattr(data_fetcher, "_yahoo_get_bars", fake_yahoo)

    start = datetime.now(UTC) - timedelta(minutes=3)
    end = datetime.now(UTC) - timedelta(minutes=1)
    result = data_fetcher.get_minute_df("AAPL", start, end)
    assert not result.empty
    assert result.attrs.get("data_provider") == "yahoo"
    assert fetch_calls["count"] == 0


def test_minute_source_override_finnhub(monkeypatch):
    monkeypatch.setenv("MINUTE_SOURCE", "finnhub")
    fetch_calls = {"count": 0}

    def _count_fetch(*args, **kwargs):
        fetch_calls["count"] += 1
        return None

    monkeypatch.setattr(data_fetcher, "_fetch_bars", _count_fetch)

    finnhub_df = _stub_frame()
    monkeypatch.setattr(
        data_fetcher,
        "_minute_df_from_finnhub",
        lambda *args, **kwargs: finnhub_df.copy(),
    )

    start = datetime.now(UTC) - timedelta(minutes=3)
    end = datetime.now(UTC) - timedelta(minutes=1)
    result = data_fetcher.get_minute_df("AAPL", start, end)
    assert not result.empty
    assert result.attrs.get("data_provider") == "finnhub"
    assert fetch_calls["count"] == 0


def test_daily_source_override_yahoo(monkeypatch):
    monkeypatch.setenv("DAILY_SOURCE", "yahoo")
    monkeypatch.setattr(data_fetcher, "should_import_alpaca_sdk", lambda: True)

    daily_df = _stub_frame()
    daily_df["timestamp"] = pd.to_datetime(daily_df["timestamp"])
    monkeypatch.setattr(data_fetcher, "_safe_backup_get_bars", lambda *args, **kwargs: daily_df.copy())

    start = datetime.now(UTC) - timedelta(days=3)
    end = datetime.now(UTC)
    result = data_fetcher.get_daily_df("AAPL", start=start, end=end)
    assert not result.empty
    assert result.attrs.get("data_provider") == "yahoo"
