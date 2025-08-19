import logging
from types import SimpleNamespace

import pytest

from ai_trading.exc import RequestException
from ai_trading.utils import http


def test_get_retries_and_logs(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    calls = {"n": 0}

    def fake_request(self, method, url, **kwargs):  # type: ignore[override]
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RequestException("boom")
        return SimpleNamespace(status_code=200, content=b"ok")

    monkeypatch.setattr(http.requests.Session, "request", fake_request)
    with caplog.at_level(logging.DEBUG):
        resp = http.get("http://example.com")
    assert resp.status_code == 200
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    debug = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert len(warnings) == 1
    assert len(debug) >= 1
