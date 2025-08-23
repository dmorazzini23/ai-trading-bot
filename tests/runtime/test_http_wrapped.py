import requests
from ai_trading.utils import http
from ai_trading.utils import HTTP_TIMEOUT, clamp_timeout


class DummyResp:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


def test_wrapped_get_retries_and_parses(monkeypatch):
    calls = []

    def fake_request(self, method, url, **kwargs):
        calls.append(kwargs)
        assert kwargs["timeout"] == clamp_timeout(None)
        if len(calls) == 1:
            raise requests.exceptions.RequestException("boom")
        return DummyResp({"ok": True})

    # Patch session.request used by wrapper
    monkeypatch.setattr(requests.Session, "request", fake_request)
    # Patch requests.get per requirement, though wrapper uses session
    monkeypatch.setattr(requests, "get", lambda url, **kw: fake_request(None, "GET", url, **kw))

    resp = http.get("https://example.com")
    assert resp.json() == {"ok": True}
    assert len(calls) == 2
