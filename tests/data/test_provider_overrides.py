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
    monkeypatch.delenv("AI_TRADING_SOURCE_REGIME_MODE", raising=False)
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
        frame = stub_df.copy()
        try:
            start_ts = start.tz_convert("UTC") if hasattr(start, "tz_convert") else start
        except Exception:
            start_ts = start
        try:
            end_ts = end.tz_convert("UTC") if hasattr(end, "tz_convert") else end
        except Exception:
            end_ts = end
        if hasattr(frame, "assign"):
            try:
                frame = frame.assign(
                    timestamp=[start_ts, start_ts + (end_ts - start_ts) / 2],
                )
            except Exception:
                frame = frame.copy()
                frame["timestamp"] = [start_ts, end_ts]
        else:
            frame = frame.copy()
            frame["timestamp"] = [start_ts, end_ts]
        return frame

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


def test_source_regime_consistency_applies_daily_override_to_minute(monkeypatch):
    monkeypatch.setenv("AI_TRADING_SOURCE_REGIME_MODE", "consistent")
    monkeypatch.setenv("DAILY_SOURCE", "yahoo")

    assert data_fetcher._env_source_override("1Min") == ("yahoo",)


def test_source_regime_consistency_respects_explicit_minute_override(monkeypatch):
    monkeypatch.setenv("AI_TRADING_SOURCE_REGIME_MODE", "consistent")
    monkeypatch.setenv("DAILY_SOURCE", "yahoo")
    monkeypatch.setenv("MINUTE_SOURCE", "alpaca")
    monkeypatch.setenv("ALPACA_DATA_FEED", "iex")

    assert data_fetcher._env_source_override("1Min") == ("alpaca_iex",)


def test_source_regime_consistency_applies_minute_override_to_daily(monkeypatch):
    monkeypatch.setenv("AI_TRADING_SOURCE_REGIME_MODE", "consistent")
    monkeypatch.setenv("MINUTE_SOURCE", "yahoo")

    assert data_fetcher._env_source_override("1Day") == ("yahoo",)
