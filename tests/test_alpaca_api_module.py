import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import alpaca_api


class DummyAPI:
    def __init__(self):
        self.calls = 0

    def submit_order(self, order_data=None):
        self.calls += 1
        if self.calls == 1:
            return types.SimpleNamespace(status_code=429)
        return types.SimpleNamespace(id=1)


def make_req():
    return types.SimpleNamespace(symbol="AAPL", qty=1, side="buy", time_in_force="day")


def test_submit_order_shadow(monkeypatch):
    api = DummyAPI()
    monkeypatch.setattr(alpaca_api, "SHADOW_MODE", True)
    result = alpaca_api.submit_order(api, make_req())
    assert result["status"] == "shadow"
    assert api.calls == 0


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
