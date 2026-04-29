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


def test_host_limits_timeout_and_session_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(http, "get_env", lambda name, default=None, **_kwargs: "2" if name == "HTTP_MAX_CONNS_PER_HOST" else default)
    http.reset_host_limit_state()

    with http.host_slot("Example.COM"):
        snapshot = http.host_limit_snapshot("example.com")
        assert snapshot["limit"] == 2
        assert snapshot["inflight"] == 1
        assert snapshot["peak"] == 1

    assert http.host_limit_snapshot("example.com")["inflight"] == 0
    assert http.reload_host_limit() == 2
    assert http._host_from_url("https://Example.COM/path") == "example.com"  # noqa: SLF001
    assert http._host_from_url("not-a-url") == "default"  # noqa: SLF001
    assert http.clamp_request_timeout((1, 2)) == (1.0, 2.0)

    monkeypatch.setattr(http, "REQUESTS_AVAILABLE", True)
    session = http.HTTPSession(timeout=(1, 3))
    sent: dict[str, object] = {}
    def fake_send(_request, **kwargs):
        sent["timeout"] = kwargs["timeout"]
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(
        session,
        "send",
        fake_send,
    )
    response = session.request("GET", "https://example.com", timeout=None)

    assert response.status_code == 200
    assert sent["timeout"] == (1.0, 3.0)


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


def test_async_fallback_wrappers_and_map_get(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(http, "REQUESTS_AVAILABLE", True)
    monkeypatch.setattr(http, "request", lambda method, url, **_kwargs: _Response(content=f"{method}:{url}".encode()))

    assert asyncio.run(http.async_request("GET", "https://example.com")).content == b"GET:https://example.com"
    assert asyncio.run(http.async_get("https://example.com")).content == b"GET:https://example.com"
    assert asyncio.run(http.async_post("https://example.com")).content == b"POST:https://example.com"
    assert asyncio.run(http.async_put("https://example.com")).content == b"PUT:https://example.com"
    assert asyncio.run(http.async_delete("https://example.com")).content == b"DELETE:https://example.com"
    assert http.get("https://example.com").content == b"GET:https://example.com"
    assert http.post("https://example.com").content == b"POST:https://example.com"
    assert http.put("https://example.com").content == b"PUT:https://example.com"
    assert http.delete("https://example.com").content == b"DELETE:https://example.com"

    monkeypatch.setattr(http, "_pool_stats", {**http._pool_stats, "workers": 1})  # noqa: SLF001
    monkeypatch.setattr(http, "_fetch_one", lambda url, timeout=None: (_ for _ in ()).throw(ValueError("bad")) if "bad" in url else (url, 200, b"ok"))
    results = http.map_get(["https://ok", "https://bad"], timeout=1)

    assert results[0][0] == ("https://ok", 200, b"ok")
    assert isinstance(results[1][1], ValueError)
    assert http.map_get([]) == []
    assert "requests" in http.pool_stats()
