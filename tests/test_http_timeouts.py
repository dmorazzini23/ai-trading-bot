"""Validate timeout behavior via the centralized HTTP abstraction."""

import inspect
import re
from unittest.mock import MagicMock

import ai_trading.utils.http as http
from pathlib import Path


def test_httpsession_sets_default_timeout(monkeypatch):
    s = http.HTTPSession(timeout=7)

    calls = {}

    def fake_get(url, **kw):
        calls["timeout"] = kw.get("timeout")
        return MagicMock(status_code=200, text="ok")

    monkeypatch.setattr(s.session, "get", fake_get)
    s.get("http://localhost/test")
    assert calls["timeout"] == 7, "HTTPSession.get must propagate default timeout"


def test_bot_engine_uses_http_abstraction():
    source = Path("ai_trading/core/bot_engine.py").read_text()
    assert (
        "from ai_trading.utils import http" in source
        or re.search(r"HTTPSession\(", source)
    ), "bot_engine should use ai_trading.utils.http (centralized timeouts/retries)."


def test_requests_can_still_be_patched_via_session(monkeypatch):
    import requests

    seen = {}

    def spy(self, method, url, **kw):
        seen["timeout_present"] = "timeout" in kw and kw["timeout"] is not None
        return MagicMock(status_code=200, text="ok")

    monkeypatch.setattr(requests.sessions.Session, "request", spy)
    requests.Session().get("http://localhost/ok", timeout=1)
    assert seen["timeout_present"] is True

