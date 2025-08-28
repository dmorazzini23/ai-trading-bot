from ai_trading.utils import http
from ai_trading.utils import HTTP_TIMEOUT, clamp_timeout
from ai_trading.exc import RequestException


class DummyResp:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


def test_wrapped_get_retries_and_parses(monkeypatch):
    calls = []

    class _Session:
        def request(self, method, url, **kwargs):
            calls.append(kwargs)
            assert kwargs["timeout"] == clamp_timeout(None)
            if len(calls) == 1:
                raise RequestException("boom")
            return DummyResp({"ok": True})

    monkeypatch.setattr(http, "_get_session", lambda: _Session())

    resp = http.get("https://example.com")
    assert resp.json() == {"ok": True}
    assert len(calls) == 2
