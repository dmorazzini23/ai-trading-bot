import types

import pytest

try:
    import alpaca_api
except Exception:
    pytest.skip("alpaca_api not available", allow_module_level=True)


class HTTPError(Exception):
    pass


class RequestException(Exception):
    pass


alpaca_api.requests = types.SimpleNamespace(
    exceptions=types.SimpleNamespace(HTTPError=HTTPError, RequestException=RequestException)
)


class DummyAPI:
    def __init__(self, to_raise=None):
        self.to_raise = to_raise or []
        self.calls = 0

    def submit_order(self, order_data=None):
        self.calls += 1
        if self.to_raise:
            exc = self.to_raise.pop(0)
            if exc is not None:
                raise exc
        return types.SimpleNamespace(id=self.calls)


class DummyReq(types.SimpleNamespace):
    pass


def test_submit_order_http_error(monkeypatch):
    api = DummyAPI([HTTPError("500"), None])
    monkeypatch.setattr(alpaca_api, "SHADOW_MODE", False)
    monkeypatch.setattr(alpaca_api.time, "sleep", lambda s: None)
    alpaca_api.submit_order(api, DummyReq())
    assert api.calls == 2


def test_submit_order_generic_retry(monkeypatch):
    api = DummyAPI([Exception("err"), None])
    monkeypatch.setattr(alpaca_api, "SHADOW_MODE", False)
    monkeypatch.setattr(alpaca_api.time, "sleep", lambda s: None)
    result = alpaca_api.submit_order(api, DummyReq())
    assert getattr(result, "id", 0) == 2
    assert api.calls == 2


def test_submit_order_fail(monkeypatch):
    api = DummyAPI([Exception("e1")] * 5)
    monkeypatch.setattr(alpaca_api, "SHADOW_MODE", False)
    monkeypatch.setattr(alpaca_api.time, "sleep", lambda s: None)
    with pytest.raises(Exception):
        alpaca_api.submit_order(api, DummyReq())
