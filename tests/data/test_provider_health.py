from __future__ import annotations

import pytest

from ai_trading.app import create_app
from ai_trading.data.fetch import (
    _record_provider_failure_event,
    _record_provider_success_event,
)
from ai_trading.telemetry import runtime_state


@pytest.fixture(autouse=True)
def _reset_provider_state():
    yield
    _record_provider_success_event()


def test_provider_degraded_state_reflected_in_health_endpoint():
    for _ in range(3):
        _record_provider_failure_event("test_failure")

    provider_state = runtime_state.observe_data_provider_state()
    assert provider_state.get("status") == "degraded"

    app = create_app()
    app.config["_ENV_VALID"] = True
    client = app.test_client()
    resp = client.get("/healthz")
    data = resp.get_json()
    assert data["data_provider"]["status"] == "degraded"
