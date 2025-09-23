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
    res = alpaca_api.submit_order("AAPL", "buy", qty=1, client=api, shadow=True)
    assert res["id"].startswith("shadow-")
    assert api.calls == 0


def test_submit_order_missing_submit(monkeypatch):
    monkeypatch.delenv("SHADOW_MODE", raising=False)
    api = object()
    with pytest.raises(AttributeError):
        alpaca_api.submit_order("AAPL", "buy", qty=1, client=api)


def test_submit_order_rate_limit(monkeypatch):
    monkeypatch.delenv("SHADOW_MODE", raising=False)
    api = DummyAPI(fail_status=429)
    with pytest.raises(Exception) as e:
        alpaca_api.submit_order("AAPL", "buy", qty=1, client=api)
    assert getattr(e.value, "status", None) == 429
    assert api.calls == 1


def test_submit_order_uses_request_object(monkeypatch):
    monkeypatch.delenv("SHADOW_MODE", raising=False)

    class _BaseRequest:
        def __init__(self, **kwargs):
            self._kwargs = kwargs
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _MarketReq(_BaseRequest):
        pass

    class _LimitReq(_BaseRequest):
        pass

    class _StopReq(_BaseRequest):
        pass

    class _StopLimitReq(_BaseRequest):
        pass

    monkeypatch.setattr(alpaca_api, "MarketOrderRequest", _MarketReq)
    monkeypatch.setattr(alpaca_api, "LimitOrderRequest", _LimitReq)
    monkeypatch.setattr(alpaca_api, "StopOrderRequest", _StopReq)
    monkeypatch.setattr(alpaca_api, "StopLimitOrderRequest", _StopLimitReq)

    class RequestClient:
        def __init__(self):
            self.calls = 0
            self.last = None

        def submit_order(self, *, order_data, idempotency_key=None):
            if isinstance(order_data, dict):  # pragma: no cover - defensive
                raise AssertionError("request object should not be a dict")
            self.calls += 1
            self.last = (order_data, idempotency_key)
            payload = {
                "id": "abc123",
                "client_order_id": getattr(order_data, "client_order_id", None),
                "symbol": getattr(order_data, "symbol", None),
                "qty": getattr(order_data, "qty", None),
                "side": getattr(order_data, "side", None),
                "time_in_force": getattr(order_data, "time_in_force", None),
            }
            return types.SimpleNamespace(id="abc123", _raw=payload)

    client = RequestClient()
    result = alpaca_api.submit_order(
        "AAPL",
        "buy",
        qty=1,
        client=client,
        idempotency_key="idem-1",
        timeout=5,
    )

    assert client.calls == 1
    request_obj, idem = client.last
    assert isinstance(request_obj, _MarketReq)
    assert request_obj.symbol == "AAPL"
    assert request_obj.qty == "1"
    assert request_obj.side == "buy"
    assert idem == "idem-1"
    assert result["client_order_id"] == "idem-1"
