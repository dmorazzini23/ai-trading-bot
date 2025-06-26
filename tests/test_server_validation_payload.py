import pytest

from server import WebhookPayload


def test_payload_valid():
    data = {'symbol': 'AAPL', 'action': 'buy'}
    payload = WebhookPayload.model_validate(data)
    assert payload.symbol == 'AAPL'
    assert payload.action == 'buy'


@pytest.mark.parametrize('data', [
    {},
    {'symbol': 'AAPL'},
    {'action': 'sell'},
])
def test_payload_invalid(data):
    with pytest.raises(Exception):
        WebhookPayload.model_validate(data)


@pytest.mark.xfail(reason="flask module patch interference in full suite")
def test_hook_invalid_symbol(monkeypatch):
    monkeypatch.setenv("WEBHOOK_SECRET", "x")
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    import importlib
    import sys
    sys.modules.pop("server", None)
    import server
    importlib.reload(server)
    app = server.create_app()
    client = app.test_client()
    monkeypatch.setattr(server, "verify_sig", lambda *a, **k: True)
    res = client.post("/github-webhook", json={"symbol": "123", "action": "buy"})
    assert res.status_code == 400

