from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

import pytest

from ai_trading.data.fetch import iex_fallback as fallback


class _Resp:
    def __init__(self, payload, *, status: int = 200, corr: str | None = None):
        self.status_code = status
        self.headers = {"Content-Type": "application/json"}
        if corr is not None:
            self.headers["x-correlation-id"] = corr
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


def test_iex_bars_empty_retries_sip(monkeypatch):
    fallback._IEX_EMPTY_COUNTS.clear()
    start, end = _dt_range()
    sess = _Session([
        _Resp({"bars": []}, corr="iex1"),
        _Resp(
            {
                "bars": [
                    {"t": "2024-01-01T00:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}
                ]
            },
            corr="sip1",
        ),
    ])
    monkeypatch.setattr(fallback, "_HTTP_SESSION", sess)
    monkeypatch.setattr(fallback, "_ALLOW_SIP", True)
    monkeypatch.setattr(fallback, "_SIP_UNAUTHORIZED", False)

    df = fallback.fetch_bars("AAPL", start, end, "1Min")

    assert len(sess.calls) == 2
    assert sess.calls[0]["feed"] == "iex"
    assert sess.calls[1]["feed"] == "sip"
    assert not df.empty
    assert fallback._IEX_EMPTY_COUNTS.get(("AAPL", "1Min"), 0) == 1


def test_iex_empty_sip_empty_logs_error(monkeypatch, caplog):
    fallback._IEX_EMPTY_COUNTS.clear()
    start, end = _dt_range()
    sess = _Session([
        _Resp({"error": "empty"}, corr="iex1"),
        _Resp({"error": "empty"}, corr="sip1"),
    ])
    monkeypatch.setattr(fallback, "_HTTP_SESSION", sess)
    monkeypatch.setattr(fallback, "_ALLOW_SIP", True)
    monkeypatch.setattr(fallback, "_SIP_UNAUTHORIZED", False)

    with caplog.at_level(logging.ERROR):
        df = fallback.fetch_bars("AAPL", start, end, "1Min")

    assert len(sess.calls) == 2
    assert sess.calls[0]["feed"] == "iex"
    assert sess.calls[1]["feed"] == "sip"
    assert getattr(df, "empty", True)
    assert any(r.message == "IEX_EMPTY_SIP_EMPTY" for r in caplog.records)
    assert fallback._IEX_EMPTY_COUNTS.get(("AAPL", "1Min"), 0) == 1


def test_skip_iex_after_empty(monkeypatch, caplog):
    fallback._IEX_EMPTY_COUNTS.clear()
    start, end = _dt_range()
    # Prime counter with an empty IEX then empty SIP
    sess1 = _Session([
        _Resp({"error": "empty"}, corr="iex1"),
        _Resp({"error": "empty"}, corr="sip1"),
    ])
    monkeypatch.setattr(fallback, "_HTTP_SESSION", sess1)
    monkeypatch.setattr(fallback, "_ALLOW_SIP", True)
    monkeypatch.setattr(fallback, "_SIP_UNAUTHORIZED", False)
    fallback.fetch_bars("AAPL", start, end, "1Min")
    assert fallback._IEX_EMPTY_COUNTS.get(("AAPL", "1Min"), 0) == 1

    # Second attempt should bypass IEX and hit SIP directly
    sess2 = _Session([
        _Resp(
            {
                "bars": [
                    {"t": "2024-01-01T00:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}
                ]
            },
            corr="sip2",
        )
    ])
    monkeypatch.setattr(fallback, "_HTTP_SESSION", sess2)

    with caplog.at_level(logging.INFO):
        df = fallback.fetch_bars("AAPL", start, end, "1Min")

    assert len(sess2.calls) == 1
    assert sess2.calls[0]["feed"] == "sip"
    assert not df.empty
    assert any(r.message == "DATA_SOURCE_FALLBACK_ATTEMPT" for r in caplog.records)
    # Counter persists to ensure future calls continue hitting SIP
    assert fallback._IEX_EMPTY_COUNTS.get(("AAPL", "1Min"), 0) == 1
