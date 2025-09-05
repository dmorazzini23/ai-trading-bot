import json
import types
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import ai_trading.data.fetch as fetch
from ai_trading.data.metrics import provider_fallback


def test_iex_empty_switches_to_sip(monkeypatch, caplog):
    symbol = "AAPL"
    start = datetime(2024, 1, 1, tzinfo=ZoneInfo("UTC"))
    end = start + timedelta(days=1)

    fetch._IEX_EMPTY_COUNTS.clear()
    fetch._IEX_EMPTY_COUNTS[(symbol, "1Min")] = fetch._IEX_EMPTY_THRESHOLD + 1
    monkeypatch.setattr(fetch, "_ALLOW_SIP", True, raising=False)
    monkeypatch.setattr(fetch, "_SIP_UNAUTHORIZED", False, raising=False)
    monkeypatch.setattr(fetch, "_window_has_trading_session", lambda *a, **k: True)
    monkeypatch.setattr(fetch, "_outside_market_hours", lambda *a, **k: False)
    monkeypatch.setattr(fetch, "is_market_open", lambda: False)

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

    allowed = [False, True]

    def sip_allowed(session, headers, timeframe):
        return allowed.pop(0)

    monkeypatch.setattr(fetch, "_sip_fallback_allowed", sip_allowed)

    before = provider_fallback.labels(
        from_provider="alpaca_iex", to_provider="alpaca_sip"
    )._value.get()
    with caplog.at_level("INFO"):
        df = fetch._fetch_bars(symbol, start, end, "1Min", feed="iex")
    after = provider_fallback.labels(
        from_provider="alpaca_iex", to_provider="alpaca_sip"
    )._value.get()

    assert feeds == ["iex", "sip"]
    assert not df.empty
    assert any(rec.message == "ALPACA_IEX_FALLBACK_SIP" for rec in caplog.records)
    assert after == before + 1
