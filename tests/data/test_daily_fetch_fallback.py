"""Regression tests for daily data fallback behaviour."""

from __future__ import annotations

from datetime import UTC, datetime
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
