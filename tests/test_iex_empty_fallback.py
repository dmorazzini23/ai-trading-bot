from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

import pytest

from ai_trading.data import fetch


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
    fetch._IEX_EMPTY_COUNTS.clear()
    start, end = _dt_range()
    sess = _Session(
        [
            _Resp({"bars": []}, corr="iex1"),
            _Resp({"bars": []}, corr="pre"),
            _Resp(
                {
                    "bars": [
                        {"t": "2024-01-01T00:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}
                    ]
                },
                corr="sip1",
            ),
        ]
    )
    monkeypatch.setattr(fetch, "_HTTP_SESSION", sess)
    monkeypatch.setattr(fetch, "_ALLOW_SIP", True)
    monkeypatch.setattr(fetch, "_SIP_UNAUTHORIZED", False)
    monkeypatch.setattr(fetch, "_FETCH_BARS_MAX_RETRIES", 0)
    monkeypatch.setattr(fetch, "_SIP_PRECHECK_DONE", False)
    monkeypatch.setattr(fetch, "_window_has_trading_session", lambda *a, **k: True)
    monkeypatch.setattr(fetch, "_outside_market_hours", lambda *a, **k: False)
    monkeypatch.setattr(fetch, "is_market_open", lambda: True)

    df = fetch._fetch_bars("AAPL", start, end, "1Min", feed="iex")

    assert len(sess.calls) == 3
    assert sess.calls[0]["feed"] == "iex"
    assert sess.calls[1]["feed"] == "sip"
    assert sess.calls[2]["feed"] == "sip"
    assert not df.empty


def test_iex_empty_falls_back_to_sip(monkeypatch, caplog):
    fetch._IEX_EMPTY_COUNTS.clear()
    start, end = _dt_range()
    sess = _Session(
        [
            _Resp({"error": "empty"}, corr="iex1"),
            _Resp({"bars": []}, corr="pre"),
            _Resp(
                {
                    "bars": [
                        {"t": "2024-01-01T00:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}
                    ]
                },
                corr="sip1",
            ),
        ]
    )
    monkeypatch.setattr(fetch, "_HTTP_SESSION", sess)
    monkeypatch.setattr(fetch, "_ALLOW_SIP", True)
    monkeypatch.setattr(fetch, "_SIP_UNAUTHORIZED", False)
    monkeypatch.setattr(fetch, "_FETCH_BARS_MAX_RETRIES", 0)
    monkeypatch.setattr(fetch, "_SIP_PRECHECK_DONE", False)

    with caplog.at_level(logging.INFO):
        df = fetch._fetch_bars("AAPL", start, end, "1Min", feed="iex")

    assert len(sess.calls) == 3
    assert sess.calls[0]["feed"] == "iex"
    assert sess.calls[1]["feed"] == "sip"
    assert sess.calls[2]["feed"] == "sip"
    assert not df.empty
    assert any(r.message == "DATA_SOURCE_FALLBACK_ATTEMPT" for r in caplog.records)
    assert fetch._IEX_EMPTY_COUNTS == {}


def test_iex_empty_sip_empty_logs_error(monkeypatch, caplog):
    fetch._IEX_EMPTY_COUNTS.clear()
    start, end = _dt_range()
    sess = _Session(
        [
            _Resp({"error": "empty"}, corr="iex1"),
            _Resp({"bars": []}, corr="pre"),
            _Resp({"error": "empty"}, corr="sip1"),
        ]
    )
    monkeypatch.setattr(fetch, "_HTTP_SESSION", sess)
    monkeypatch.setattr(fetch, "_ALLOW_SIP", True)
    monkeypatch.setattr(fetch, "_SIP_UNAUTHORIZED", False)
    monkeypatch.setattr(fetch, "_FETCH_BARS_MAX_RETRIES", 0)
    monkeypatch.setattr(fetch, "_SIP_PRECHECK_DONE", False)

    with caplog.at_level(logging.ERROR):
        df = fetch._fetch_bars("AAPL", start, end, "1Min", feed="iex")

    assert len(sess.calls) == 3
    assert sess.calls[0]["feed"] == "iex"
    assert sess.calls[1]["feed"] == "sip"
    assert sess.calls[2]["feed"] == "sip"
    assert df is None or getattr(df, "empty", True)
    assert any(r.message == "IEX_EMPTY_SIP_EMPTY" for r in caplog.records)
    assert fetch._IEX_EMPTY_COUNTS.get(("AAPL", "1Min"), 0) == 1


def test_get_minute_df_skips_iex_after_empty(monkeypatch, caplog):
    fetch._IEX_EMPTY_COUNTS.clear()
    start, end = _dt_range()
    # First attempt: IEX empty then SIP empty to prime the counter
    sess1 = _Session([
        _Resp({"error": "empty"}, corr="iex1"),
        _Resp({"bars": []}, corr="pre"),
        _Resp({"error": "empty"}, corr="sip1"),
    ])
    monkeypatch.setattr(fetch, "_HTTP_SESSION", sess1)
    monkeypatch.setattr(fetch, "_ALLOW_SIP", True)
    monkeypatch.setattr(fetch, "_SIP_UNAUTHORIZED", False)
    monkeypatch.setattr(fetch, "_FETCH_BARS_MAX_RETRIES", 0)
    monkeypatch.setattr(fetch, "_SIP_PRECHECK_DONE", False)
    fetch.get_minute_df("AAPL", start, end)
    assert fetch._IEX_EMPTY_COUNTS.get(("AAPL", "1Min"), 0) == 1

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
    monkeypatch.setattr(fetch, "_HTTP_SESSION", sess2)

    with caplog.at_level(logging.INFO):
        df = fetch.get_minute_df("AAPL", start, end)

    assert len(sess2.calls) == 1
    assert sess2.calls[0]["feed"] == "sip"
    assert not df.empty
    assert any(r.message == "DATA_SOURCE_FALLBACK_ATTEMPT" for r in caplog.records)
    assert fetch._IEX_EMPTY_COUNTS.get(("AAPL", "1Min"), 0) == 0

