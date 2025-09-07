from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

import pytest

from ai_trading.data.fetch import iex_fallback
from ai_trading.data.fetch import _IEX_EMPTY_COUNTS


class _Resp:
    def __init__(self, payload, *, status: int = 200):
        self.status_code = status
        self.headers = {"Content-Type": "application/json"}
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.get = self._get

    def _get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(params)
        return self._responses.pop(0)


def _dt_range():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=1)
    return start, end


def setup_module(module):  # pragma: no cover - test helper
    _IEX_EMPTY_COUNTS.clear()


def test_iex_empty_switches_to_sip(monkeypatch, caplog):
    _IEX_EMPTY_COUNTS.clear()
    start, end = _dt_range()
    sess = _Session([
        _Resp({"bars": []}),
        _Resp({"bars": [{"t": "2024-01-01T00:00:00Z"}]}),
    ])
    monkeypatch.setattr(iex_fallback, "_ALLOW_SIP", True)
    monkeypatch.setattr(iex_fallback, "_SIP_UNAUTHORIZED", False)
    with caplog.at_level(logging.INFO):
        df = iex_fallback.fetch_bars("AAPL", start, end, "1Min", session=sess)
    assert [c["feed"] for c in sess.calls] == ["iex", "sip"]
    assert not df.empty
    assert _IEX_EMPTY_COUNTS == {}
    assert any(r.message == "DATA_SOURCE_FALLBACK_ATTEMPT" for r in caplog.records)


def test_both_feeds_empty_logs_error(monkeypatch, caplog):
    _IEX_EMPTY_COUNTS.clear()
    start, end = _dt_range()
    sess = _Session([
        _Resp({"bars": []}),
        _Resp({"bars": []}),
    ])
    monkeypatch.setattr(iex_fallback, "_ALLOW_SIP", True)
    monkeypatch.setattr(iex_fallback, "_SIP_UNAUTHORIZED", False)
    with caplog.at_level(logging.ERROR):
        df = iex_fallback.fetch_bars("AAPL", start, end, "1Min", session=sess)
    assert [c["feed"] for c in sess.calls] == ["iex", "sip"]
    assert getattr(df, "empty", True)
    assert _IEX_EMPTY_COUNTS.get(("AAPL", "1Min"), 0) == 1
    assert any(r.message == "IEX_EMPTY_SIP_EMPTY" for r in caplog.records)


def test_skip_iex_after_threshold(monkeypatch, caplog):
    _IEX_EMPTY_COUNTS.clear()
    start, end = _dt_range()
    # First call primes the counter
    sess1 = _Session([
        _Resp({"bars": []}),
        _Resp({"bars": []}),
    ])
    monkeypatch.setattr(iex_fallback, "_ALLOW_SIP", True)
    monkeypatch.setattr(iex_fallback, "_SIP_UNAUTHORIZED", False)
    iex_fallback.fetch_bars("AAPL", start, end, "1Min", session=sess1)
    assert _IEX_EMPTY_COUNTS.get(("AAPL", "1Min"), 0) == 1

    # Second call should bypass IEX and hit SIP directly
    sess2 = _Session([
        _Resp({"bars": [{"t": "2024-01-01T00:00:00Z"}]}),
    ])
    with caplog.at_level(logging.INFO):
        df = iex_fallback.fetch_bars("AAPL", start, end, "1Min", session=sess2)
    assert len(sess2.calls) == 1
    assert sess2.calls[0]["feed"] == "sip"
    assert not df.empty
    assert _IEX_EMPTY_COUNTS.get(("AAPL", "1Min"), 0) == 0
    assert any(r.message == "DATA_SOURCE_FALLBACK_ATTEMPT" for r in caplog.records)
