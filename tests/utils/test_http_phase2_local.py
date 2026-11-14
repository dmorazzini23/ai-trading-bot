from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from ai_trading.utils import http


class _Response:
    def __init__(self, status_code: int = 200, payload=None, text: str = "ok", content: bytes = b"ok") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.content = content

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload




def test_request_and_request_json_retry_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"request": 0, "json": 0}

    class Session:
        _timeout = 5

        def request(self, method: str, url: str, **kwargs):
            calls["request"] += 1
            if calls["request"] == 1:
                raise http.RequestException("temporary")
            return _Response(payload={"method": method, "url": url, "timeout": kwargs.get("timeout")})

    monkeypatch.setattr(http, "REQUESTS_AVAILABLE", True)
    monkeypatch.setattr(http, "_get_session", lambda: Session())
    monkeypatch.setattr(http, "_retry_config", lambda: (2, 0.0, 0.0, 0.0))
    monkeypatch.setattr(http, "retry_call", lambda func, **_kwargs: func() if calls["request"] else func())

    with pytest.raises(http.RequestException):
        http.request("GET", "https://example.com")

    calls["request"] = 1
    assert http.request("POST", "https://example.com").status_code == 200

    class JsonSession:
        def request(self, *_args, **_kwargs):
            calls["json"] += 1
            if calls["json"] == 1:
                return _Response(status_code=500, payload={"retry": True})
            if calls["json"] == 2:
                return _Response(status_code=200, payload=ValueError("bad json"), text="plain")
            return _Response(status_code=200, payload=[1, 2])

    monkeypatch.setattr(http, "_get_session", lambda: JsonSession())
    monkeypatch.setattr(http, "sleep", lambda _seconds: None)

    assert http.request_json("GET", "https://example.com", retries=2) == {"text": "plain"}
    assert http.request_json("GET", "https://example.com", retries=1) == {"data": [1, 2]}


def test_request_json_raises_on_terminal_retry_status(monkeypatch: pytest.MonkeyPatch) -> None:
    class JsonSession:
        def request(self, *_args, **_kwargs):
            return _Response(status_code=429, payload={"error": "rate limited"})

    monkeypatch.setattr(http, "REQUESTS_AVAILABLE", True)
    monkeypatch.setattr(http, "_get_session", lambda: JsonSession())
    monkeypatch.setattr(http, "sleep", lambda _seconds: None)

    with pytest.raises(http.RequestException, match="status 429"):
        http.request_json("GET", "https://example.com", retries=1)


