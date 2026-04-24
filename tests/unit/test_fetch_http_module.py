import pytest
import ai_trading.data.fetch.http as fetch_http


class _Session:
    def __init__(self):
        self.calls = 0
        self.last_kwargs = None

    def get(self, url, **kwargs):  # noqa: ARG002
        self.calls += 1
        self.last_kwargs = dict(kwargs)
        return {}


def test_unauthorized_sip_raises_before_request(monkeypatch: pytest.MonkeyPatch):
    session = _Session()
    monkeypatch.setattr(fetch_http, "_HTTP_SESSION", session, raising=False)
    monkeypatch.setattr(fetch_http, "_SIP_UNAUTHORIZED", True, raising=False)

    with pytest.raises(ValueError, match="sip_unauthorized"):
        fetch_http.get("https://example.com", feed="sip")

    assert session.calls == 0


def test_get_injects_default_timeout(monkeypatch: pytest.MonkeyPatch):
    session = _Session()
    monkeypatch.setattr(fetch_http, "_HTTP_SESSION", session, raising=False)
    monkeypatch.setattr(fetch_http, "_SIP_UNAUTHORIZED", False, raising=False)

    fetch_http.get("https://example.com", feed="iex")

    assert session.last_kwargs is not None
    assert session.last_kwargs["timeout"] is not None
