from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from datetime import UTC

import pytest

import ai_trading.data.fetch as df
from ai_trading.utils.timefmt import isoformat_z


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


def test_rate_limit_fallback_success(monkeypatch: pytest.MonkeyPatch, capmetrics: list[Rec]):
    start, end = _ts_window()
    responses = [Resp(429, {}), Resp(200, _bars_payload(start))]

    def fake_get(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(df.requests, "get", fake_get, raising=True)
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


def test_timeout_fallback_success(monkeypatch: pytest.MonkeyPatch, capmetrics: list[Rec]):
    start, end = _ts_window()
    events: list[object] = [df.Timeout("boom"), Resp(200, _bars_payload(start))]

    def fake_get(*args, **kwargs):
        ev = events.pop(0)
        if isinstance(ev, Exception):
            raise ev
        return ev

    monkeypatch.setattr(df.requests, "get", fake_get, raising=True)
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


def test_unauthorized_sip_fallback_success(
    monkeypatch: pytest.MonkeyPatch, capmetrics: list[Rec]
):
    start, end = _ts_window()
    responses = [Resp(403, {}), Resp(200, _bars_payload(start))]

    def fake_get(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(df.requests, "get", fake_get, raising=True)
    out = df._fetch_bars("AAPL", start, end, "1Min", feed="sip")
    assert not out.empty
    names = [r.name for r in capmetrics]
    assert names == [
        "data.fetch.unauthorized",
        "data.fetch.fallback_attempt",
        "data.fetch.success",
    ]
    assert capmetrics[0].tags["feed"] == "sip"
    assert capmetrics[1].tags["feed"] == "iex"


def test_empty_payload_fallback_success(
    monkeypatch: pytest.MonkeyPatch, capmetrics: list[Rec]
):
    start, end = _ts_window()
    responses = [Resp(200, {"bars": []}), Resp(200, _bars_payload(start))]

    def fake_get(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(df.requests, "get", fake_get, raising=True)
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

