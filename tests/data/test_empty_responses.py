import pandas as pd
from datetime import datetime, timedelta, UTC

import ai_trading.data.fetch as fetch


def _make_df():
    return pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 2, tzinfo=UTC)],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1],
        }
    )


class _Resp:
    status_code = 200
    text = "{\"bars\":[]}" 
    headers = {"Content-Type": "application/json"}

    def json(self):
        return {"bars": []}


class _RespNonEmpty(_Resp):
    text = "{\"bars\":[{\"t\":\"2024-01-02T15:30:00Z\",\"o\":1.0,\"h\":1.0,\"l\":1.0,\"c\":1.0,\"v\":1}]}"

    def json(self):
        return {
            "bars": [
                {
                    "t": "2024-01-02T15:30:00Z",
                    "o": 1.0,
                    "h": 1.0,
                    "l": 1.0,
                    "c": 1.0,
                    "v": 1,
                }
            ]
        }


def test_alpaca_empty_responses_trigger_backup(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    calls = {"count": 0, "feeds": []}

    def fake_get(*args, **kwargs):
        calls["count"] += 1
        params = kwargs.get("params") or {}
        calls["feeds"].append(params.get("feed"))
        if params.get("feed") == "iex":
            return _Resp()
        return _RespNonEmpty()

    monkeypatch.setattr(fetch._HTTP_SESSION, "get", fake_get)
    monkeypatch.setattr(fetch, "_ENABLE_HTTP_FALLBACK", True)
    monkeypatch.setattr(fetch, "provider_priority", lambda: ["alpaca_iex", "alpaca_sip", "yahoo"])
    monkeypatch.setattr(fetch, "max_data_fallbacks", lambda: 2)
    monkeypatch.setattr(fetch, "_symbol_exists", lambda symbol: True)
    monkeypatch.setattr(fetch, "_window_has_trading_session", lambda start, end: True)
    monkeypatch.setattr(fetch, "_outside_market_hours", lambda start, end: False)
    monkeypatch.setattr(fetch, "is_market_open", lambda: True)
    monkeypatch.setattr(fetch, "_yahoo_get_bars", lambda *args, **kwargs: _make_df())

    start = datetime(2024, 1, 2, 15, 30, tzinfo=UTC)
    end = start + timedelta(minutes=1)
    df = fetch._fetch_bars("AAPL", start, end, "1Min", feed="iex")

    assert calls["count"] == 2
    assert calls["feeds"] == ["iex", "sip"]
    assert not df.empty
