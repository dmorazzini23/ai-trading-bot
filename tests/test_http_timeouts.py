"""Validate timeout behavior via the centralized HTTP abstraction."""

import re
from pathlib import Path
from unittest.mock import MagicMock

from ai_trading.utils import http

pytest_plugins = ("tests.watchdog_ext",)


def _get_session_type():
    """Resolve the requests Session type robustly."""  # AI-AGENT-REF: mirror plugin lookup
    import importlib

    import requests

    Session = getattr(requests, "Session", None)
    if Session is None:
        Session = getattr(importlib.import_module("requests.sessions"), "Session", None)
    return Session


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
    assert "from ai_trading.utils import http" in source or re.search(
        r"HTTPSession\(", source
    ), "bot_engine should use ai_trading.utils.http (centralized timeouts/retries)."


def test_requests_can_still_be_patched_via_session():
    """
    Validate that our test plugin injects a default timeout into raw requests.Session calls
    even when the caller does not pass `timeout`.
    We avoid patching internals and instead mount a spy adapter to observe kwargs.
    """
    from requests.adapters import HTTPAdapter
    from requests.models import Response

    Session = _get_session_type()
    assert Session is not None, "requests Session type must be importable for this test"

    seen = {}

    class SpyAdapter(HTTPAdapter):
        def send(self, request, **kwargs):
            # capture timeout kw passed down by Session.request (via our plugin)
            seen["timeout"] = kwargs.get("timeout")
            resp = Response()
            resp.status_code = 200
            resp._content = b"ok"
            resp.url = request.url
            return resp

    s = Session()
    s.mount("http://", SpyAdapter())
    s.mount("https://", SpyAdapter())

    # Do not pass timeout -> plugin should inject default
    r = s.get(
        "http://localhost/_probe_ok"
    )  # AI-AGENT-REF: trigger plugin default timeout
    assert r.status_code == 200
    assert (
        seen.get("timeout") is not None
    ), "Default timeout should be injected by the plugin"
