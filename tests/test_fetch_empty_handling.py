import json
import logging
import types
from datetime import UTC, datetime, timedelta

import pytest

from ai_trading.data import fetch


class _Resp:
    def __init__(self, payload):
        self.status_code = 200
        self.headers = {"Content-Type": "application/json"}
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _Session:
    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls = 0
        self.get = self._get

    def _get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        return _Resp(self._payloads.pop(0))


def _dt_range():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=1)
    return start, end


def test_warn_on_empty_when_market_open(monkeypatch, caplog):
    monkeypatch.setattr(fetch, "_window_has_trading_session", lambda *a, **k: True)
    start, end = _dt_range()
    sess = _Session([{ "bars": []} for _ in range(4)])
    monkeypatch.setattr(fetch, "_HTTP_SESSION", sess)
    monkeypatch.setattr(fetch, "_SIP_UNAUTHORIZED", True)
    monkeypatch.setattr(fetch, "is_market_open", lambda: True)
    monkeypatch.setattr(fetch, "_empty_should_emit", lambda *a, **k: True)
    monkeypatch.setattr(fetch, "_empty_record", lambda *a, **k: 1)
    monkeypatch.setattr(fetch, "_empty_classify", lambda **k: logging.WARNING)
    monkeypatch.setattr(fetch, "_outside_market_hours", lambda *a, **k: False)

    elapsed = 0.0
    delays: list[float] = []

    def _monotonic() -> float:
        return elapsed

    def _sleep(sec: float) -> None:
        nonlocal elapsed
        elapsed += sec
        delays.append(sec)

    monkeypatch.setattr(fetch, "time", types.SimpleNamespace(monotonic=_monotonic, sleep=_sleep))

    with caplog.at_level(logging.DEBUG):
        out = fetch._fetch_bars("AAPL", start, end, "1Min")

    assert out is None
    assert sess.calls == 4
    retry_logs = [r for r in caplog.records if r.message == "RETRY_EMPTY_BARS"]
    assert [r.attempt for r in retry_logs] == [1, 2, 3]
    assert [r.total_elapsed for r in retry_logs] == [0, 1, 3]
    assert delays == [1, 2, 4]
    assert any(r.message == "EMPTY_DATA" and r.levelno == logging.WARNING for r in caplog.records)
    assert any(r.message == "ALPACA_FETCH_RETRY_LIMIT" for r in caplog.records)


def test_silent_fallback_when_market_closed(monkeypatch, caplog):
    monkeypatch.setattr(fetch, "_window_has_trading_session", lambda *a, **k: True)
    start, end = _dt_range()
    payloads = [
        {"bars": []},
        {"bars": [{"t": "2024-01-01T00:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}]},
    ]
    sess = _Session(payloads)
    monkeypatch.setattr(fetch, "_HTTP_SESSION", sess)
    monkeypatch.setattr(fetch, "_SIP_UNAUTHORIZED", False)
    monkeypatch.setattr(fetch, "is_market_open", lambda: False)

    monkeypatch.setattr(
        fetch,
        "time",
        types.SimpleNamespace(monotonic=lambda: 0.0, sleep=lambda _s: None),
    )

    with caplog.at_level(logging.INFO):
        df = fetch._fetch_bars("AAPL", start, end, "1Min")

    assert not df.empty
    assert all(r.message != "EMPTY_DATA" for r in caplog.records)


def test_skip_retry_outside_market_hours(monkeypatch, caplog):
    monkeypatch.setattr(fetch, "_window_has_trading_session", lambda *a, **k: True)
    start, end = _dt_range()
    sess = _Session([{"bars": []}])
    monkeypatch.setattr(fetch, "_HTTP_SESSION", sess)
    monkeypatch.setattr(fetch, "_SIP_UNAUTHORIZED", True)
    monkeypatch.setattr(fetch, "is_market_open", lambda: False)
    monkeypatch.setattr(fetch, "_outside_market_hours", lambda *a, **k: True)

    with caplog.at_level(logging.INFO):
        out = fetch._fetch_bars("AAPL", start, end, "1Min")

    assert out is None
    assert sess.calls == 1
    assert any(r.message == "ALPACA_FETCH_MARKET_CLOSED" for r in caplog.records)
