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
        self.get = self._get

    def _get(self, url, params=None, headers=None, timeout=None):
        idx = self.calls
        self.calls += 1
        return _Resp(self._payloads[idx], self._corr_ids[idx])


def _dt_range():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=1)
    return start, end


def test_skip_log_when_retries_exhausted(monkeypatch, caplog):
    start, end = _dt_range()
    monkeypatch.setattr(fetch, "_window_has_trading_session", lambda *a, **k: True)
    payloads = [{"bars": []}, {"bars": []}, {"bars": []}]
    corr_ids = ["id1", "id2", "id3"]
    sess = _Session(payloads, corr_ids)
    monkeypatch.setattr(fetch, "_HTTP_SESSION", sess)
    monkeypatch.setattr(fetch, "_SIP_UNAUTHORIZED", True)
    monkeypatch.setattr(fetch, "is_market_open", lambda: True)
    monkeypatch.setattr(fetch, "_sip_fallback_allowed", lambda *a, **k: False)
    monkeypatch.setattr(fetch, "_outside_market_hours", lambda *a, **k: False)
    monkeypatch.setattr(fetch, "_FETCH_BARS_MAX_RETRIES", 2)
    monkeypatch.setattr(fetch, "_ENABLE_HTTP_FALLBACK", False)
    monkeypatch.setattr(fetch, "max_data_fallbacks", lambda: 0)
    monkeypatch.setattr(fetch, "_backup_get_bars", lambda *a, **k: None)
    monkeypatch.setattr(fetch, "_symbol_exists", lambda *a, **k: True)
    monkeypatch.setattr(
        fetch,
        "time",
        types.SimpleNamespace(monotonic=lambda: 0.0, sleep=lambda _s: None),
    )

    calls = []

    def _log_fetch_attempt(provider, *, status=None, error=None, **extra):
        calls.append((status, error, extra))

    monkeypatch.setattr(fetch, "log_fetch_attempt", _log_fetch_attempt)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(fetch.EmptyBarsError) as exc:
            fetch._fetch_bars("AAPL", start, end, "1Min", feed="iex")

    assert "alpaca empty response" in str(exc.value)
    assert sess.calls == 2
    assert len(calls) == 2
    assert calls[-1][2]["remaining_retries"] == fetch._FETCH_BARS_MAX_RETRIES - 1
    assert all(c[2]["remaining_retries"] >= 0 for c in calls)
    assert all(c[1] == "empty" for c in calls)
