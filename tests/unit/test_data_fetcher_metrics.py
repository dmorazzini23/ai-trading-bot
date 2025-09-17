from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from datetime import UTC

import pytest

import ai_trading.data.fetch as df
from ai_trading.utils.timefmt import isoformat_z

pd = pytest.importorskip("pandas")


@dataclass
class Rec:
    name: str
    tags: dict
    value: float


@pytest.fixture
def capmetrics(monkeypatch: pytest.MonkeyPatch):
    bucket: list[Rec] = []

    def record(name: str, value: float = 1.0, tags: dict | None = None):
        bucket.append(Rec(name, tags or {}, value))

    # Patch the real metrics hook directly (no shim)
    monkeypatch.setattr(df.metrics, "incr", record, raising=False)
    return bucket


def _bars_payload(ts: dt.datetime) -> dict:
    # Format timestamp with ai_trading.utils.timefmt helper
    ts_s = isoformat_z(ts)
    return {"bars": [{"t": ts_s, "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}]}


class Resp:
    def __init__(self, status_code: int, payload: dict | None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = "" if payload is None else "{}"
        self.headers = {"Content-Type": "application/json"}

    def json(self) -> dict:
        return self._payload


def _ts_window() -> tuple[dt.datetime, dt.datetime]:
    start = dt.datetime(2024, 1, 1, tzinfo=UTC)
    end = start + dt.timedelta(minutes=1)
    return start, end


@pytest.fixture(autouse=True)
def _force_window(monkeypatch):
    monkeypatch.setattr(df, "_window_has_trading_session", lambda *a, **k: True)


def test_rate_limit_fallback_success(monkeypatch: pytest.MonkeyPatch, capmetrics: list[Rec]):
    monkeypatch.setattr(df, "_SIP_UNAUTHORIZED", False, raising=False)
    start, end = _ts_window()
    responses = [Resp(429, {}), Resp(200, _bars_payload(start))]

    def fake_get(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(df._HTTP_SESSION, "get", fake_get, raising=False)
    monkeypatch.setattr(df.requests, "get", fake_get, raising=False)
    out = df._fetch_bars("AAPL", start, end, "1Min", feed="iex")
    assert not out.empty
    names = [r.name for r in capmetrics]
    assert names == [
        "data.fetch.rate_limited",
        "data.fetch.fallback_attempt",
        "data.fetch.success",
    ]
    assert capmetrics[0].tags["feed"] == "iex"
    assert capmetrics[1].tags["feed"] == "sip"
    assert capmetrics[2].tags["feed"] == "sip"


def test_rate_limit_no_retry_when_sip_unauthorized(
    monkeypatch: pytest.MonkeyPatch, capmetrics: list[Rec]
):
    monkeypatch.setattr(df, "_SIP_UNAUTHORIZED", True, raising=False)
    start, end = _ts_window()
    calls = {"count": 0}

    def fake_get(*args, **kwargs):
        calls["count"] += 1
        return Resp(429, {})

    monkeypatch.setattr(df._HTTP_SESSION, "get", fake_get, raising=False)
    monkeypatch.setattr(df.requests, "get", fake_get, raising=False)

    with pytest.raises(ValueError, match="rate_limited"):
        df._fetch_bars("AAPL", start, end, "1Min", feed="iex")

    assert calls["count"] == 1
    names = [r.name for r in capmetrics]
    assert names == ["data.fetch.rate_limited"]
    assert capmetrics[0].tags["feed"] == "iex"


def test_timeout_fallback_success(monkeypatch: pytest.MonkeyPatch, capmetrics: list[Rec]):
    monkeypatch.setattr(df, "_SIP_UNAUTHORIZED", False, raising=False)
    start, end = _ts_window()
    events: list[object] = [df.Timeout("boom"), Resp(200, _bars_payload(start))]

    def fake_get(*args, **kwargs):
        ev = events.pop(0)
        if isinstance(ev, Exception):
            raise ev
        return ev

    monkeypatch.setattr(df._HTTP_SESSION, "get", fake_get, raising=False)
    monkeypatch.setattr(df.requests, "get", fake_get, raising=False)
    out = df._fetch_bars("AAPL", start, end, "1Min", feed="iex")
    assert not out.empty
    names = [r.name for r in capmetrics]
    assert names == [
        "data.fetch.timeout",
        "data.fetch.fallback_attempt",
        "data.fetch.success",
    ]
    assert capmetrics[0].tags["feed"] == "iex"
    assert capmetrics[1].tags["feed"] == "sip"


def test_unauthorized_sip_returns_empty(
    monkeypatch: pytest.MonkeyPatch, capmetrics: list[Rec]
):
    monkeypatch.setattr(df, "_SIP_UNAUTHORIZED", False, raising=False)
    start, end = _ts_window()
    responses = [Resp(403, {})]

    def fake_get(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(df._HTTP_SESSION, "get", fake_get, raising=False)
    monkeypatch.setattr(df.requests, "get", fake_get, raising=False)
    monkeypatch.setattr(df, "_backup_get_bars", lambda *a, **k: pd.DataFrame())
    out = df._fetch_bars("AAPL", start, end, "1Min", feed="sip")
    assert out is None or out.empty
    names = [r.name for r in capmetrics]
    assert names == ["data.fetch.unauthorized"]
    assert capmetrics[0].tags["feed"] == "sip"
    assert df._SIP_UNAUTHORIZED is True


def test_empty_payload_fallback_success(
    monkeypatch: pytest.MonkeyPatch, capmetrics: list[Rec]
):
    monkeypatch.setattr(df, "_SIP_UNAUTHORIZED", False, raising=False)
    start, end = _ts_window()
    responses = [Resp(200, {"bars": []}), Resp(200, _bars_payload(start))]

    def fake_get(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(df._HTTP_SESSION, "get", fake_get, raising=False)
    monkeypatch.setattr(df.requests, "get", fake_get, raising=False)
    out = df._fetch_bars("AAPL", start, end, "1Min", feed="iex")
    assert not out.empty
    names = [r.name for r in capmetrics]
    assert names == [
        "data.fetch.empty",
        "data.fetch.fallback_attempt",
        "data.fetch.success",
    ]
    assert capmetrics[0].tags["feed"] == "iex"
    assert capmetrics[1].tags["feed"] == "sip"

