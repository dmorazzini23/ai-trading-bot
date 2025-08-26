import builtins
from ai_trading.app import create_app


def test_health_endpoint_handles_import_error(monkeypatch):
    original_import = builtins.__import__

    def fail_alpaca(name, *args, **kwargs):
        if name == "ai_trading.alpaca_api":
            raise ImportError("boom")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_alpaca)
    app = create_app()
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("ok") is False
    assert "boom" in data.get("error", "")
