"""Regression tests for daily data fallback behaviour."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import sys
import types

import pytest

from ai_trading.data import fetch as fetch_module
from ai_trading.utils.lazy_imports import load_pandas

pytest.importorskip("pandas")


def test_get_daily_df_uses_backup_when_columns_missing(monkeypatch):
    pd = load_pandas()
    assert pd is not None

    monkeypatch.setattr(fetch_module, "should_import_alpaca_sdk", lambda: True)
    monkeypatch.setattr(
        fetch_module,
        "get_settings",
        lambda: types.SimpleNamespace(
            backup_data_provider="yahoo",
            logging_dedupe_ttl_s=0,
        ),
    )

    backup_calls: dict[str, tuple] = {}

    def _fake_backup_get_bars(symbol, start, end, interval):
        backup_calls["args"] = (symbol, start, end, interval)
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-02"], utc=True),
                "open": [1.0],
                "high": [1.5],
                "low": [0.5],
                "close": [1.25],
                "volume": [100],
            }
        )
        df.attrs["fallback_provider"] = "yahoo"
        return df

    monkeypatch.setattr(fetch_module, "_backup_get_bars", _fake_backup_get_bars)

    alpaca_stub = types.ModuleType("ai_trading.alpaca_api")

    def _raise_missing(*_args, **_kwargs):
        raise fetch_module.MissingOHLCVColumnsError("missing columns")

    alpaca_stub.get_bars_df = _raise_missing
    monkeypatch.setitem(sys.modules, "ai_trading.alpaca_api", alpaca_stub)

    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 3, tzinfo=UTC)

    result = fetch_module.get_daily_df("AAPL", start=start, end=end)

    assert "args" in backup_calls
    symbol, start_dt, end_dt, interval = backup_calls["args"]
    assert symbol == "AAPL"
    assert interval == "1d"
    assert start_dt <= start
    assert end_dt >= end
    assert list(result.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert not result.empty


def test_get_daily_df_normalizes_yahoo_regular_market_schema(monkeypatch):
    pd = load_pandas()
    assert pd is not None

    monkeypatch.setattr(fetch_module, "should_import_alpaca_sdk", lambda: True)
    monkeypatch.setattr(
        fetch_module,
        "get_settings",
        lambda: types.SimpleNamespace(
            backup_data_provider="yahoo",
            logging_dedupe_ttl_s=0,
        ),
    )

    def _yahoo_regular_market_frame():
        return pd.DataFrame(
            {
                "regularMarketTime": pd.to_datetime(["2024-01-02"], utc=True),
                "regularMarketOpen": [1.0],
                "regularMarketDayHigh": [1.5],
                "regularMarketDayLow": [0.5],
                "regularMarketPrice": [1.25],
                "regularMarketPreviousClose": [1.2],
                "regularMarketVolume": [100],
            }
        )

    def _fake_backup_get_bars(symbol, start, end, interval):
        df = _yahoo_regular_market_frame()
        df.attrs["fallback_provider"] = "yahoo"
        return df

    monkeypatch.setattr(fetch_module, "_backup_get_bars", _fake_backup_get_bars)

    alpaca_stub = types.ModuleType("ai_trading.alpaca_api")

    def _raise_missing(*_args, **_kwargs):
        raise fetch_module.MissingOHLCVColumnsError("missing columns")

    alpaca_stub.get_bars_df = _raise_missing
    monkeypatch.setitem(sys.modules, "ai_trading.alpaca_api", alpaca_stub)

    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 3, tzinfo=UTC)

    result = fetch_module.get_daily_df("AAPL", start=start, end=end)

    assert list(result.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert result["open"].iloc[0] == 1.0
    assert result["high"].iloc[0] == 1.5
    assert result["low"].iloc[0] == 0.5
    assert result["close"].iloc[0] == 1.25
    assert result["volume"].iloc[0] == 100


def test_ensure_ohlcv_schema_handles_yahoo_premarket_payload():
    pd = load_pandas()
    assert pd is not None

    payload = [
        {
            "t": "2024-01-02T09:30:00Z",
            "preMarketOpen": 1.0,
            "preMarketDayHigh": 1.5,
            "preMarketDayLow": 0.5,
            "preMarketPrice": 1.25,
            "preMarketVolume": 100,
        }
    ]

    frame = pd.DataFrame(payload)
    fetch_module._attach_payload_metadata(
        frame,
        payload=payload,
        provider="yahoo",
        feed="yahoo",
        timeframe="1Min",
        symbol="AAPL",
    )

    ensured = fetch_module.ensure_ohlcv_schema(frame, source="yahoo", frequency="1Min")

    assert list(ensured.columns[:6]) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    first = ensured.iloc[0]
    assert pytest.approx(first["open"]) == payload[0]["preMarketOpen"]
    assert pytest.approx(first["high"]) == payload[0]["preMarketDayHigh"]
    assert pytest.approx(first["low"]) == payload[0]["preMarketDayLow"]
    assert pytest.approx(first["close"]) == payload[0]["preMarketPrice"]
    assert pytest.approx(first["volume"]) == payload[0]["preMarketVolume"]


def test_get_minute_df_handles_yahoo_premarket_backup(monkeypatch):
    pd = load_pandas()
    assert pd is not None

    start = datetime(2024, 1, 2, 9, 30, tzinfo=UTC)
    end = start + timedelta(minutes=1)

    monkeypatch.setattr(fetch_module, "_ensure_pandas", lambda: pd)
    monkeypatch.setattr(fetch_module, "pd", pd)
    monkeypatch.setattr(fetch_module, "_window_has_trading_session", lambda *a, **k: True)
    monkeypatch.setattr(fetch_module, "_outside_market_hours", lambda *a, **k: False)
    monkeypatch.setattr(fetch_module, "is_market_open", lambda: True)
    monkeypatch.setattr(fetch_module, "_post_process", lambda df, *_, **__: df)
    monkeypatch.setattr(fetch_module, "_verify_minute_continuity", lambda df, *_, **__: df)
    monkeypatch.setattr(
        fetch_module,
        "_repair_rth_minute_gaps",
        lambda df, *_, **__: (df, {"expected": 0, "missing_after": 0, "gap_ratio": 0.0}, False),
    )
    monkeypatch.setattr(fetch_module, "mark_success", lambda *a, **k: None)
    monkeypatch.setattr(fetch_module, "_mark_fallback", lambda *a, **k: None)
    monkeypatch.setattr(fetch_module, "_incr", lambda *a, **k: None)
    monkeypatch.setattr(
        fetch_module.provider_monitor,
        "active_provider",
        lambda primary, backup: backup,
    )
    monkeypatch.setattr(fetch_module.provider_monitor, "record_switchover", lambda *a, **k: None)

    def _primary_fail(*_args, **_kwargs):
        raise RuntimeError("primary down")

    monkeypatch.setattr(fetch_module, "_fetch_bars", _primary_fail)

    payload = [
        {
            "t": "2024-01-02T09:30:00Z",
            "preMarketOpen": 1.0,
            "preMarketDayHigh": 1.5,
            "preMarketDayLow": 0.5,
            "preMarketPrice": 1.25,
            "preMarketVolume": 100,
        }
    ]

    def _fake_backup_get_bars(symbol, start_dt, end_dt, interval):
        frame = pd.DataFrame(payload)
        fetch_module._attach_payload_metadata(
            frame,
            payload=payload,
            provider="yahoo",
            feed="yahoo",
            timeframe="1Min",
            symbol=symbol,
        )
        frame.attrs["data_provider"] = "yahoo"
        frame.attrs["data_feed"] = "yahoo"
        return frame

    monkeypatch.setattr(fetch_module, "_backup_get_bars", _fake_backup_get_bars)

    # Reset provider monitor state so fallback activation decisions are deterministic.
    fetch_module.provider_monitor.threshold = 1
    fetch_module.provider_monitor.cooldown = 0
    fetch_module.provider_monitor.fail_counts.clear()
    fetch_module.provider_monitor.disabled_until.clear()
    fetch_module.provider_monitor.disable_counts.clear()
    fetch_module.provider_monitor.outage_start.clear()

    result = fetch_module.get_minute_df("AAPL", start, end)

    assert result is not None
    assert not result.empty
    assert list(result.columns[:6]) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    first = result.iloc[0]
    assert pytest.approx(first["open"]) == payload[0]["preMarketOpen"]
    assert pytest.approx(first["high"]) == payload[0]["preMarketDayHigh"]
    assert pytest.approx(first["low"]) == payload[0]["preMarketDayLow"]
    assert pytest.approx(first["close"]) == payload[0]["preMarketPrice"]
    assert pytest.approx(first["volume"]) == payload[0]["preMarketVolume"]
