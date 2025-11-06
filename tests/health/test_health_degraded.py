from types import SimpleNamespace

from ai_trading.health import HealthCheck
from ai_trading.telemetry import runtime_state


def test_health_degraded_payload_includes_diagnostics() -> None:
    runtime_state.update_data_provider_state(
        primary="alpaca",
        active=None,
        status="down",
        reason="safe_mode",
        cooldown_sec=120.0,
        gap_ratio_recent=0.15,
    )
    runtime_state.update_broker_status(connected=True, status="reachable")

    ctx = SimpleNamespace(service="ai-trading")
    hc = HealthCheck(ctx=ctx)
    client = hc.app.test_client()

    response = client.get("/healthz")
    assert response.status_code == 200
    payload = response.get_json()

    assert payload["status"] == "degraded"
    assert payload["ok"] is False
    assert payload["reason"] == "safe_mode"
    assert payload["cooldown_seconds_remaining"] == 120.0
    assert payload["gap_ratio_recent"] == 0.15
    assert "timestamp" in payload

    provider = payload["primary_data_provider"]
    assert provider["status"] == "down"
    assert provider["cooldown_seconds_remaining"] == 120.0
    assert provider["gap_ratio_recent"] == 0.15

