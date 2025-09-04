import json
import logging
import types
from datetime import UTC, datetime, timedelta

import pytest

from ai_trading.data import fetch


class _Resp:
    def __init__(self, payload, corr_id):
        self.status_code = 200
        self.headers = {"Content-Type": "application/json", "x-request-id": corr_id}
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _Session:
    def __init__(self, payloads, corr_ids):
        self._payloads = list(payloads)
        self._corr_ids = list(corr_ids)
        self.calls = 0
        self.get = self._get  # ensure "get" in __dict__ for use_session_get

    def _get(self, url, params=None, headers=None, timeout=None):
        idx = self.calls
        self.calls += 1
        return _Resp(self._payloads[idx], self._corr_ids[idx])


def _dt_range():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=1)
    return start, end


def test_single_retry_and_warning(monkeypatch, caplog):
    start, end = _dt_range()
    payloads = [
        {"bars": []},
        {
            "bars": [
                {"t": "2024-01-01T00:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}
            ]
        },
    ]
    corr_ids = ["id1", "id2"]
    sess = _Session(payloads, corr_ids)
    monkeypatch.setattr(fetch, "_HTTP_SESSION", sess)
    monkeypatch.setattr(fetch, "is_market_open", lambda: True)
    monkeypatch.setattr(fetch, "_sip_fallback_allowed", lambda *a, **k: False)
    monkeypatch.setattr(fetch, "_outside_market_hours", lambda *a, **k: False)

    sleep_called = {}
    monkeypatch.setattr(
        fetch,
        "time",
        types.SimpleNamespace(sleep=lambda s: sleep_called.setdefault("delay", s)),
    )

    calls = []

    def _log_fetch_attempt(provider, *, status=None, error=None, **extra):
        calls.append((status, error, extra))

    monkeypatch.setattr(fetch, "log_fetch_attempt", _log_fetch_attempt)

    with caplog.at_level(logging.WARNING):
        df = fetch._fetch_bars("AAPL", start, end, "1Min", feed="iex")

    assert not df.empty
    assert sess.calls == 2
    assert sleep_called.get("delay") is not None
    # Only one warning about empty data
    assert sum(r.message == "EMPTY_DATA" for r in caplog.records) == 1
    # log_fetch_attempt called once per request
    assert len(calls) == 2
    assert calls[0][1] == "empty"
    assert calls[0][2]["correlation_id"] == "id1"
    assert "previous_correlation_id" not in calls[0][2]
    assert calls[1][1] is None
    assert calls[1][2]["correlation_id"] == "id2"
    assert calls[1][2]["previous_correlation_id"] == "id1"
