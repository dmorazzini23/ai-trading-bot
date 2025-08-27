from __future__ import annotations


import json
from datetime import UTC, datetime, timedelta

import pytest

pd = pytest.importorskip("pandas")

import ai_trading.data.fetch as df


class _Resp:
    def __init__(self, status_code: int, payload: dict | None = None, content_type: str = "application/json"):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = {"Content-Type": content_type}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


def _bars_payload(ts_iso: str) -> dict:
    return {
        "bars": [
            {
                "t": ts_iso,
                "o": 10.0,
                "h": 11.0,
                "l": 9.5,
                "c": 10.5,
                "v": 1000,
            }
        ]
    }


def _dt_range(minutes: int = 5):
    end = datetime.now(UTC).replace(microsecond=0)
    start = end - timedelta(minutes=minutes)
    return start, end


@pytest.mark.parametrize("status_first", [401, 403])
def test_sip_unauthorized_returns_empty(monkeypatch: pytest.MonkeyPatch, status_first: int):
    monkeypatch.setattr(df, "_SIP_UNAUTHORIZED", False, raising=False)
    calls = {"count": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        calls["count"] += 1
        return _Resp(status_first, payload={"message": "auth required"})

    monkeypatch.setattr(df, "requests", type("R", (), {"get": staticmethod(fake_get)}))

    start, end = _dt_range(2)
    out = df.get_bars("TEST", timeframe="1Min", start=start, end=end, feed="sip", adjustment="raw")
    assert isinstance(out, pd.DataFrame) and out.empty
    assert calls["count"] == 1
    assert df._SIP_UNAUTHORIZED is True


def test_sip_fallback_skipped_when_marked_unauthorized(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(df, "_SIP_UNAUTHORIZED", True, raising=False)
    calls = {"count": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        calls["count"] += 1
        return _Resp(429, payload={"message": "rate limit"})

    monkeypatch.setattr(df, "requests", type("R", (), {"get": staticmethod(fake_get)}))

    start, end = _dt_range(2)
    with pytest.raises(ValueError, match="rate_limited"):
        df._fetch_bars("TEST", start, end, "1Min", feed="iex")
    assert calls["count"] == 1


def test_timeout_triggers_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(df, "_SIP_UNAUTHORIZED", False, raising=False)
    calls = {"count": 0}

    class _Timeout(df.Timeout):
        pass

    def fake_get(url, params=None, headers=None, timeout=None):
        calls["count"] += 1
        feed = (params or {}).get("feed")
        if calls["count"] == 1 and feed == "sip":
            raise _Timeout("timeout")
        ts_iso = datetime.now(UTC).isoformat()
        return _Resp(200, payload=_bars_payload(ts_iso))

    monkeypatch.setattr(df, "requests", type("R", (), {"get": staticmethod(fake_get)}))

    start, end = _dt_range(2)
    out = df.get_bars("TEST", timeframe="1Min", start=start, end=end, feed="sip", adjustment="raw")
    assert isinstance(out, pd.DataFrame) and not out.empty
    assert calls["count"] >= 2


def test_429_rate_limit_triggers_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(df, "_SIP_UNAUTHORIZED", False, raising=False)
    calls = {"count": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        calls["count"] += 1
        feed = (params or {}).get("feed")
        ts_iso = datetime.now(UTC).isoformat()
        if calls["count"] == 1 and feed == "sip":
            return _Resp(429, payload={"message": "rate limit"})
        return _Resp(200, payload=_bars_payload(ts_iso))

    monkeypatch.setattr(df, "requests", type("R", (), {"get": staticmethod(fake_get)}))

    start, end = _dt_range(2)
    out = df.get_bars("TEST", timeframe="1Min", start=start, end=end, feed="sip", adjustment="raw")
    assert isinstance(out, pd.DataFrame) and not out.empty
    assert calls["count"] >= 2


def test_empty_bars_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(df, "_SIP_UNAUTHORIZED", False, raising=False)
    calls = {"count": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        calls["count"] += 1
        ts_iso = datetime.now(UTC).isoformat()
        if calls["count"] == 1:
            return _Resp(200, payload={"bars": []})
        return _Resp(200, payload=_bars_payload(ts_iso))

    monkeypatch.setattr(df, "requests", type("R", (), {"get": staticmethod(fake_get)}))

    start, end = _dt_range(2)
    out = df.get_bars("TEST", timeframe="1Min", start=start, end=end, feed="iex", adjustment="raw")
    assert isinstance(out, pd.DataFrame) and not out.empty
    assert calls["count"] >= 2
