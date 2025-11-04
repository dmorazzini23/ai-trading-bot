from types import SimpleNamespace

from ai_trading.health import HealthCheck
from ai_trading.telemetry import runtime_state


def test_health_endpoint_reports_runtime_state():
    runtime_state.update_data_provider_state(primary="alpaca", active="yahoo", backup="yahoo", using_backup=True, reason="test")
    runtime_state.update_quote_status(allowed=False, reason="missing_bid_ask", status="stale", synthetic=False)
    runtime_state.update_broker_status(connected=False, last_error="timeout")

    ctx = SimpleNamespace(service="ai-trading")
    hc = HealthCheck(ctx=ctx)
    client = hc.app.test_client()
    response = client.get("/healthz")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["service"] == "ai-trading"
    assert payload["fallback_active"] is True
    provider_info = payload["primary_data_provider"]
    assert provider_info["active"] == "yahoo"
    assert payload["quotes_status"]["status"] == "stale"
    assert payload["broker_connectivity"]["connected"] is False


def test_health_endpoint_reports_degraded_provider_state():
    runtime_state.update_data_provider_state(primary="alpaca", active="finnhub", using_backup=True, status="degraded")
    ctx = SimpleNamespace(service="ai-trading")
    hc = HealthCheck(ctx=ctx)
    client = hc.app.test_client()

    response = client.get("/healthz")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "degraded"
    assert payload["ok"] is True
    provider_section = payload["primary_data_provider"]
    assert provider_section["status"] == "degraded"
    assert payload["fallback_active"] is True
