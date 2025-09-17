import json
import types
from datetime import UTC, datetime, timedelta

import pytest

import ai_trading.data.fetch as fetch


def _dt_range():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=1)
    return start, end


def test_error_empty_switches_to_backup(monkeypatch):
    pd = pytest.importorskip("pandas")
    start, end = _dt_range()
    fetch._ALPACA_EMPTY_ERROR_COUNTS.clear()
    monkeypatch.setattr(fetch, "_ALLOW_SIP", False, raising=False)
    monkeypatch.setattr(fetch, "_ENABLE_HTTP_FALLBACK", False, raising=False)
    monkeypatch.setattr(fetch, "max_data_fallbacks", lambda: 0)
    monkeypatch.setattr(fetch, "provider_priority", lambda: ["alpaca_iex"])
    monkeypatch.setattr(fetch, "_preferred_feed_failover", lambda: [])
    monkeypatch.setattr(fetch, "_ALPACA_EMPTY_ERROR_THRESHOLD", 2, raising=False)
    monkeypatch.setattr(fetch, "_FETCH_BARS_MAX_RETRIES", 1, raising=False)
    monkeypatch.setattr(fetch, "_window_has_trading_session", lambda *a, **k: True)
    monkeypatch.setattr(fetch, "_outside_market_hours", lambda *a, **k: False)
    monkeypatch.setattr(fetch, "is_market_open", lambda: True)
    monkeypatch.setattr(
        fetch,
        "get_settings",
        lambda: types.SimpleNamespace(backup_data_provider="yahoo"),
    )

    class _Resp:
        def __init__(self, corr_id: str):
            self.status_code = 200
            self.headers = {"Content-Type": "application/json", "x-request-id": corr_id}
            self.text = json.dumps({"error": "empty"})

        def json(self):
            return {"error": "empty"}

    class _Requests:
        def __init__(self):
            self.calls = 0

        def get(self, url, params=None, headers=None, timeout=None):
            self.calls += 1
            return _Resp(f"id{self.calls}")

    req = _Requests()
    monkeypatch.setattr(fetch, "requests", req)
    monkeypatch.setattr(fetch._HTTP_SESSION, "get", req.get)

    backup_calls = {"count": 0}

    def fake_backup(symbol, start, end, interval):
        backup_calls["count"] += 1
        return pd.DataFrame(
            [
                {
                    "timestamp": start,
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "volume": 1,
                }
            ]
        )

    monkeypatch.setattr(fetch, "_backup_get_bars", fake_backup)

    out1 = fetch._fetch_bars("AAPL", start, end, "1Min", feed="iex")
    assert out1 is not None and not out1.empty
    assert req.calls == 1
    assert backup_calls["count"] == 1

    out2 = fetch._fetch_bars("AAPL", start, end, "1Min", feed="iex")
    assert out2 is not None and not out2.empty
    assert req.calls == 1
    assert backup_calls["count"] == 2

    assert ("AAPL", "1Min") not in fetch._ALPACA_EMPTY_ERROR_COUNTS
