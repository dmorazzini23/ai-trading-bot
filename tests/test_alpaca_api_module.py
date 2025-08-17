import types

import pytest

try:
    from ai_trading import alpaca_api  # AI-AGENT-REF: canonical import
except Exception:
    pytest.skip("alpaca_api not available", allow_module_level=True)


class DummyAPI:
    def __init__(self):
        self.calls = 0

    def submit_order(self, **order_data):
        self.calls += 1
        if self.calls == 1:
            return types.SimpleNamespace(status_code=429)
        return types.SimpleNamespace(id=1, **order_data)


def make_req():
    return types.SimpleNamespace(symbol="AAPL", qty=1, side="buy", time_in_force="day")


def test_submit_order_shadow(monkeypatch):
    api = DummyAPI()
    monkeypatch.setattr(alpaca_api, "SHADOW_MODE", True)
    result = alpaca_api.submit_order(api, make_req())
    assert result.id == "dry-run"
    assert result.status == "accepted"
    assert api.calls == 0


def test_submit_order_missing_submit(monkeypatch):
    monkeypatch.setattr(alpaca_api, "SHADOW_MODE", False)
    api = object()  # lacks submit_order
    result = alpaca_api.submit_order(api, make_req())
    assert result.id == "dry-run"
    assert result.status == "accepted"


def test_submit_order_rate_limit(monkeypatch):
    api = DummyAPI()
    monkeypatch.setattr(alpaca_api, "SHADOW_MODE", False)
    monkeypatch.setattr(
        alpaca_api.requests,
        "exceptions",
        types.SimpleNamespace(HTTPError=Exception),
        raising=False,
    )
    sleeps = []
    monkeypatch.setattr(alpaca_api.time, "sleep", lambda s: sleeps.append(s))
    result = alpaca_api.submit_order(api, make_req())
    assert getattr(result, "id", None) == 1
    assert api.calls == 2
    assert sleeps
