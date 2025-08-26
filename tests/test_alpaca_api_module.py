import types
import pytest

from ai_trading import alpaca_api


class DummyAPI:
    def __init__(self, fail_status: int | None = None):
        self.calls = 0
        self.fail_status = fail_status

    def submit_order(self, **order_data):
        self.calls += 1
        if self.fail_status and self.calls == 1:
            err = Exception("fail")
            err.status = self.fail_status
            raise err
        return types.SimpleNamespace(id="123", **order_data)


def test_submit_order_shadow(monkeypatch):
    api = DummyAPI()
    monkeypatch.setenv("SHADOW_MODE", "1")
    res = alpaca_api.submit_order("AAPL", 1, "buy", client=api)
    assert res["id"].startswith("shadow-")
    assert api.calls == 0


def test_submit_order_missing_submit(monkeypatch):
    monkeypatch.delenv("SHADOW_MODE", raising=False)
    api = object()
    with pytest.raises(AttributeError):
        alpaca_api.submit_order("AAPL", 1, "buy", client=api)


def test_submit_order_rate_limit(monkeypatch):
    monkeypatch.delenv("SHADOW_MODE", raising=False)
    api = DummyAPI(fail_status=429)
    with pytest.raises(Exception) as e:
        alpaca_api.submit_order("AAPL", 1, "buy", client=api)
    assert getattr(e.value, "status", None) == 429
    assert api.calls == 1
