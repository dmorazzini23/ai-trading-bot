import json
import types
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

import ai_trading.data.fetch as fetch
import ai_trading.net.http as http
from ai_trading.data.fetch.metrics import inc_provider_fallback

pytest.importorskip("pandas")


@pytest.mark.parametrize(
    ("preferred_feeds", "provider_order"),
    [
        (("sip",), ["alpaca_iex", "alpaca_sip"]),
        ((), ["alpaca_iex"]),
    ],
    ids=["preferred_list", "no_preference"],
)
def test_iex_empty_switches_to_sip(monkeypatch, caplog, preferred_feeds, provider_order):
    symbol = "AAPL"
    start = datetime(2024, 1, 1, tzinfo=ZoneInfo("UTC"))
    end = start + timedelta(days=1)

    fetch._IEX_EMPTY_COUNTS.clear()
    fetch._FEED_OVERRIDE_BY_TF.clear()
    fetch._FEED_FAILOVER_ATTEMPTS.clear()
    fetch._FEED_SWITCH_HISTORY.clear()
    fetch._FEED_SWITCH_LOGGED.clear()
    fetch._cycle_feed_override.clear()
    fetch._IEX_EMPTY_COUNTS[(symbol, "1Min")] = fetch._IEX_EMPTY_THRESHOLD + 1
    monkeypatch.setenv("TESTING", "1")
    monkeypatch.setenv("ALPACA_API_KEY", "dummy")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "dummy")
    monkeypatch.setattr(fetch, "_ALLOW_SIP", True, raising=False)
    monkeypatch.setattr(fetch, "_SIP_UNAUTHORIZED", False, raising=False)
    monkeypatch.setattr(fetch, "_window_has_trading_session", lambda *a, **k: True)
    monkeypatch.setattr(fetch, "_outside_market_hours", lambda *a, **k: False)
    monkeypatch.setattr(fetch, "is_market_open", lambda: False)
    monkeypatch.setattr(fetch, "alpaca_feed_failover", lambda: preferred_feeds)
    monkeypatch.setattr(fetch, "provider_priority", lambda: provider_order)

    feeds = []

    def fake_get(url, params=None, headers=None, timeout=None):
        feeds.append(params.get("feed"))
        if params.get("feed") == "iex":
            data = {"bars": []}
        else:
            data = {"bars": [{"t": "2024-01-01T00:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}]}
        return types.SimpleNamespace(
            status_code=200,
            text=json.dumps(data),
            headers={"Content-Type": "application/json"},
            json=lambda: data,
        )

    monkeypatch.setattr(fetch.requests, "get", fake_get)
    monkeypatch.setattr(http.requests, "get", fake_get, raising=False)

    allowed = [False, True]

    def sip_allowed(session, headers, timeframe):
        return allowed.pop(0)

    monkeypatch.setattr(fetch, "_sip_fallback_allowed", sip_allowed)

    before = inc_provider_fallback("alpaca_iex", "alpaca_sip")
    with caplog.at_level("INFO"):
        df = fetch._fetch_bars(symbol, start, end, "1Min", feed="iex")
    after = inc_provider_fallback("alpaca_iex", "alpaca_sip")

    assert feeds == ["iex", "sip"]
    assert not df.empty
    assert any(rec.message == "ALPACA_IEX_FALLBACK_SIP" for rec in caplog.records)
    assert fetch._FEED_OVERRIDE_BY_TF[(symbol, "1Min")] == "sip"
    assert fetch._cycle_feed_override.get(symbol) == "sip"
    assert "sip" in fetch._FEED_FAILOVER_ATTEMPTS.get((symbol, "1Min"), set())
    assert after == before + 2
