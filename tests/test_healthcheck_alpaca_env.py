from __future__ import annotations

from types import SimpleNamespace

from ai_trading.health import HealthCheck


def test_healthcheck_reports_alpaca_credential_presence_from_env(monkeypatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "test-key-123")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret-123")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    hc = HealthCheck(ctx=SimpleNamespace(service="ai-trading"))
    client = hc.app.test_client()
    response = client.get("/healthz")
    payload = response.get_json() if hasattr(response, "get_json") else response

    assert isinstance(payload, dict)
    alpaca = payload.get("alpaca", {})
    assert alpaca.get("has_key") is True
    assert alpaca.get("has_secret") is True
    assert alpaca.get("base_url") == "https://paper-api.alpaca.markets"
    assert alpaca.get("paper") is True
