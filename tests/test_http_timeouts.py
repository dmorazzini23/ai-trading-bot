"""Validate timeout behavior via the centralized HTTP abstraction."""

from unittest.mock import MagicMock

import pytest
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
    if http.REQUESTS_AVAILABLE:
        captured: dict[str, float] = {}

        def fake_request(self, method, url, **kw):  # type: ignore[override]
            captured.update(kw)
            return MagicMock(status_code=200, text="ok")

        monkeypatch.setattr(http.requests.Session, "request", fake_request, raising=True)
        s.get("http://localhost/test")
        assert captured["timeout"] == 7, "HTTPSession.get must propagate default timeout"
    else:
        with pytest.raises(RuntimeError):
            s.get("http://localhost/test")


@pytest.mark.skipif(not http.REQUESTS_AVAILABLE, reason="requests not installed")
def test_request_uses_session_default(monkeypatch):
    captured: dict[str, float] = {}

    def fake_request(self, method, url, **kwargs):  # type: ignore[override]
        captured.update(kwargs)
        return MagicMock(status_code=200)

    monkeypatch.setattr(http.requests.Session, "request", fake_request, raising=True)
    sess = http.HTTPSession(timeout=5)
    monkeypatch.setattr(http, "_get_session", lambda: sess)
    http.request("GET", "http://unit.test")
    assert captured["timeout"] == 5

    captured.clear()
    http.request("GET", "http://unit.test", timeout=1.23)
    assert captured["timeout"] == 1.23


def test_request_function_errors_without_requests():
    if http.REQUESTS_AVAILABLE:
        pytest.skip("requires requests missing")
    with pytest.raises(RuntimeError):
        http.request("GET", "http://example.com")


@pytest.mark.skipif(not http.REQUESTS_AVAILABLE, reason="requests not installed")
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
