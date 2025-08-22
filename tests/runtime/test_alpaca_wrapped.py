import pytest

pytestmark = pytest.mark.alpaca

pytest.importorskip("alpaca")

from ai_trading.broker.alpaca import AlpacaBroker, APIError


class DummyClient:
    def submit_order(self, **_):  # type: ignore[override]
        return {"ok": True}


def test_submit_order_retries(monkeypatch):
    client = DummyClient()
    calls = {"n": 0}

    def flaky(**_):
        calls["n"] += 1
        if calls["n"] == 1:
            raise APIError("boom")
        return {"ok": True}

    monkeypatch.setattr(client, "submit_order", flaky)
    broker = AlpacaBroker(client)
    resp = broker.submit_order(symbol="AAPL", qty=1, side="buy")
    assert resp == {"ok": True}
    assert calls["n"] == 2
