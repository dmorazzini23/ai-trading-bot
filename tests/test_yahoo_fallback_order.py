import json
import types
import datetime as dt
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
    fetch._SIP_UNAUTHORIZED = False
    fetch._ALLOW_SIP = True
    fetch._ENABLE_HTTP_FALLBACK = True

    def fake_get(url, params=None, headers=None, timeout=None):
        data = {"bars": []}
        return types.SimpleNamespace(
            status_code=200,
            text=json.dumps(data),
            headers={"Content-Type": "application/json"},
            json=lambda: data,
        )

    monkeypatch.setattr(fetch.requests, "get", fake_get)

    called = {}

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

    assert called.get("yahoo")
    assert not df.empty
    assert after == before + 1
    assert fo.FALLBACK_ORDER.get("yahoo")
