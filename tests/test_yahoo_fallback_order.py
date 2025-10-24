import json
import types
import datetime as dt
from typing import Any

import pytest

pd = pytest.importorskip("pandas")

import ai_trading.data.fetch as fetch
from ai_trading.data.fetch.metrics import inc_provider_fallback
import ai_trading.data.fetch.fallback_order as fo


def test_yahoo_used_after_two_alpaca_failures(monkeypatch):
    symbol = "AAPL"
    start = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
    end = start + dt.timedelta(minutes=1)

    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ENABLE_HTTP_FALLBACK", "1")
    monkeypatch.setenv("AI_TRADING_YAHOO_FAILURE_THRESHOLD", "2")
    monkeypatch.delenv("TESTING", raising=False)
    fetch._SIP_UNAUTHORIZED = False
    fetch._ALLOW_SIP = True
    fetch._ENABLE_HTTP_FALLBACK = True

    called = {}

    class StubSession:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def get(self, url, params=None, headers=None, timeout=None):
            called["http_get"] = called.get("http_get", 0) + 1
            self.calls.append({
                "url": url,
                "params": dict(params or {}),
                "headers": dict(headers or {}),
            })
            data = {"bars": []}
            return types.SimpleNamespace(
                status_code=200,
                text=json.dumps(data),
                headers={"Content-Type": "application/json"},
                json=lambda: data,
            )

    session = StubSession()
    monkeypatch.setattr(fetch, "_HTTP_SESSION", session)

    def fake_yahoo(symbol, start, end, interval):
        called["yahoo"] = True
        return pd.DataFrame(
            {
                "timestamp": [pd.Timestamp(start)],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [0],
            }
        )

    monkeypatch.setattr(fetch, "_yahoo_get_bars", fake_yahoo)
    fo.reset()

    before = inc_provider_fallback("alpaca_sip", "yahoo")

    df = fetch._fetch_bars(symbol, start, end, "1Min", feed="iex")

    after = inc_provider_fallback("alpaca_sip", "yahoo")

    assert called.get("http_get")
    assert called.get("yahoo")
    assert not df.empty
    assert after == before + 1
    assert fo.FALLBACK_ORDER["yahoo"]
    assert fo.FALLBACK_PROVIDERS and fo.FALLBACK_PROVIDERS[-1] == "yahoo"
    assert fo.FALLBACK_SYMBOLS and fo.FALLBACK_SYMBOLS[-1] == symbol


def test_yahoo_fallback_suppressed_until_threshold(monkeypatch):
    symbol = "MSFT"
    start = dt.datetime(2024, 2, 1, 14, 30, tzinfo=dt.UTC)
    end = start + dt.timedelta(minutes=1)

    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ENABLE_HTTP_FALLBACK", "1")
    monkeypatch.setenv("AI_TRADING_YAHOO_FAILURE_THRESHOLD", "5")
    fetch._ENABLE_HTTP_FALLBACK = True
    fetch._SIP_UNAUTHORIZED = False
    fetch._ALLOW_SIP = True
    fetch._ALPACA_SYMBOL_FAILURES.clear()
    fetch._ALPACA_FAILURE_EVENTS.clear()
    fo.reset()

    class StubSession:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def get(self, url, params=None, headers=None, timeout=None):
            self.calls.append({
                "url": url,
                "params": dict(params or {}),
                "headers": dict(headers or {}),
            })
            payload = {"bars": []}
            return types.SimpleNamespace(
                status_code=200,
                text=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                json=lambda: payload,
            )

    def fake_yahoo(symbol, start, end, interval):
        return pd.DataFrame()

    session = StubSession()
    monkeypatch.setattr(fetch, "_HTTP_SESSION", session)
    monkeypatch.setattr(fetch, "_yahoo_get_bars", fake_yahoo)

    df = fetch._fetch_bars(symbol, start, end, "1Min", feed="iex")

    provider_attr = getattr(df, "attrs", {}).get("fallback_provider") if df is not None else None
    assert provider_attr != "yahoo"
    assert df is None or getattr(df, "empty", True)
    assert "yahoo" not in fo.FALLBACK_ORDER
