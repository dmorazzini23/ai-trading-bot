import json
import logging
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

    def get(self, url, params=None, headers=None, timeout=None):
        return _Resp(self._payloads.pop(0))


def _dt_range():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=1)
    return start, end


def test_warn_on_empty_when_market_open(monkeypatch, caplog):
    start, end = _dt_range()
    sess = _Session([{"bars": []}])
    monkeypatch.setattr(fetch, "_HTTP_SESSION", sess)
    monkeypatch.setattr(fetch, "_SIP_UNAUTHORIZED", True)
    monkeypatch.setattr(fetch, "is_market_open", lambda: True)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(ValueError, match="empty_bars"):
            fetch._fetch_bars("AAPL", start, end, "1Min")

    assert any(r.message == "EMPTY_DATA" and r.levelno == logging.WARNING for r in caplog.records)


def test_silent_fallback_when_market_closed(monkeypatch, caplog):
    start, end = _dt_range()
    payloads = [
        {"bars": []},
        {"bars": [{"t": "2024-01-01T00:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}]},
    ]
    sess = _Session(payloads)
    monkeypatch.setattr(fetch, "_HTTP_SESSION", sess)
    monkeypatch.setattr(fetch, "_SIP_UNAUTHORIZED", False)
    monkeypatch.setattr(fetch, "is_market_open", lambda: False)

    with caplog.at_level(logging.INFO):
        df = fetch._fetch_bars("AAPL", start, end, "1Min")

    assert not df.empty
    assert all(r.message != "EMPTY_DATA" for r in caplog.records)
