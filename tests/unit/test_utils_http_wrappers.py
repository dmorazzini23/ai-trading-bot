import pytest

from ai_trading.utils import http


@pytest.mark.parametrize(
    "func, method",
    [
        (http.get, "GET"),
        (http.post, "POST"),
        (http.put, "PUT"),
        (http.delete, "DELETE"),
    ],
)
def test_wrappers_delegate_request(func, method, monkeypatch):
    captured = {}

    def fake_request(m, url, **kwargs):
        captured["args"] = (m, url)
        captured["kwargs"] = kwargs
        return "response"

    monkeypatch.setattr(http, "request", fake_request)
    url = "https://example.com"
    kwargs = {"timeout": 1, "headers": {"X": "y"}}
    resp = func(url, **kwargs)
    assert resp == "response"
    assert captured["args"] == (method, url)
    assert captured["kwargs"] == kwargs
