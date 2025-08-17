import types

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
    monkeypatch.setattr(alpaca_api, "SHADOW_MODE", True)
    res = alpaca_api.submit_order(api, symbol="AAPL", qty=1, side="buy")
    assert res.success
    assert res.status == "shadow"
    assert api.calls == 0


def test_submit_order_missing_submit(monkeypatch):
    monkeypatch.setattr(alpaca_api, "SHADOW_MODE", False)
    api = object()
    res = alpaca_api.submit_order(api, symbol="AAPL", qty=1, side="buy")
    assert res.success
    assert res.status == "shadow"


def test_submit_order_rate_limit(monkeypatch):
    monkeypatch.setattr(alpaca_api, "SHADOW_MODE", False)
    api = DummyAPI(fail_status=429)
    res = alpaca_api.submit_order(api, symbol="AAPL", qty=1, side="buy")
    assert not res.success
    assert res.retryable
    assert res.status == 429
    assert api.calls == 1
