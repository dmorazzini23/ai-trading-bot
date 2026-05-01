from __future__ import annotations

import json

import ai_trading.governance.promotion as promotion_mod
import ai_trading.services.control_plane as control_plane_mod

from ai_trading.app import create_app
from ai_trading.operator_presets import PresetValidationError, build_plan
from ai_trading.telemetry import runtime_state


def _operator_headers(
    *,
    operator_id: str = "ops@example.com",
    token: str = "test-operator-token",
) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "X-AI-Trading-Operator-Id": operator_id,
    }


def _post_json(client, path: str, payload: dict[str, object]):
    post = getattr(client, "post", None)
    if callable(post):
        return post(path, json=payload, headers=_operator_headers())
    return client.open(path, method="POST", json=payload, headers=_operator_headers())


def _delete(client, path: str):
    delete = getattr(client, "delete", None)
    if callable(delete):
        return delete(path, headers=_operator_headers())
    return client.open(path, method="DELETE", headers=_operator_headers())


def _get(client, path: str):
    get = getattr(client, "get", None)
    if callable(get):
        return get(path, headers=_operator_headers())
    return client.open(path, method="GET", headers=_operator_headers())


def _configure_operator_auth(
    monkeypatch,
    *,
    operator_id: str = "ops@example.com",
    token: str = "test-operator-token",
) -> None:
    monkeypatch.setenv("AI_TRADING_OPERATOR_TOKEN_MAP", json.dumps({operator_id: token}))


def test_operator_presets_endpoint(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    _configure_operator_auth(monkeypatch)
    app = create_app()
    client = app.test_client()

    response = _get(client, "/operator/presets")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    names = {item["name"] for item in payload["presets"]}
    assert {"conservative", "balanced", "aggressive"} <= names


def test_operator_plan_endpoint_returns_default_plan(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    _configure_operator_auth(monkeypatch)
    app = create_app()
    client = app.test_client()

    response = _get(client, "/operator/plan")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["plan"]["name"] == "balanced"


def test_operator_control_plane_snapshot_endpoint(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    _configure_operator_auth(monkeypatch)
    app = create_app()
    client = app.test_client()

    response = _get(client, "/operator/control-plane")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    snapshot = payload["snapshot"]
    assert snapshot["service"] == "ai-trading"
    assert "rollout" in snapshot
    assert "broker_health" in snapshot
    assert "execution_quality" in snapshot
    assert "manual_overrides" in snapshot
    assert "governance" in snapshot
    execution_quality = snapshot["execution_quality"]
    assert "submit_reject_reasons_top" in execution_quality
    assert "cancel_reasons_top" in execution_quality
    assert "realized_slippage_decomposition" in execution_quality
    assert "event_outcomes_by_scope" in execution_quality
    governance = snapshot["governance"]
    assert "latest_promotion_approval" in governance
    assert "latest_champion_challenger_scorecard" in governance
    assert "latest_rollback_audit" in governance
    services = snapshot["services"]
    assert "signal" in services
    assert "execution" in services
    assert "governance" in services


def test_operator_control_plane_services_endpoint(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    _configure_operator_auth(monkeypatch)
    app = create_app()
    client = app.test_client()

    response = _get(client, "/operator/control-plane/services")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["section"] == "services"
    assert payload["data"]["risk_approval"]["owner"] == "ai_trading.services.risk_approval"
    assert payload["data"]["execution"]["boundary_type"] == "facade"
    assert "submit_runtime" in payload["data"]["execution"]["canonical_runtime_owner"][0]
    assert payload["data"]["portfolio"]["boundary_type"] == "facade"
    assert payload["data"]["reconciliation"]["boundary_type"] == "facade"


def test_operator_control_plane_services_uses_runtime_state_fast_path(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    _configure_operator_auth(monkeypatch)
    runtime_state.reset_all_states()
    runtime_state.update_data_provider_state(status="healthy", active="alpaca")
    runtime_state.update_broker_status(
        connected=True,
        status="connected",
        open_orders_count=0,
        positions_count=2,
    )
    runtime_state.update_service_status(status="ready", phase="active")

    def _unexpected_snapshot(*, service_name: str):
        raise AssertionError(f"unexpected full snapshot build for {service_name}")

    monkeypatch.setattr(control_plane_mod, "build_control_plane_snapshot", _unexpected_snapshot)
    app = create_app()
    client = app.test_client()

    try:
        response = _get(client, "/operator/control-plane/services")
    finally:
        runtime_state.reset_all_states()

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["section"] == "services"
    assert payload["data"]["execution"]["status"] == "connected"
    assert payload["data"]["signal"]["status"] == "healthy"
    assert payload["data"]["portfolio"]["status"] == "ready"


def test_operator_control_plane_open_orders_uses_runtime_state_fast_path(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    _configure_operator_auth(monkeypatch)
    runtime_state.reset_broker_status()
    runtime_state.update_broker_status(
        connected=True,
        latency_ms=12.5,
        status="reachable",
        open_orders_count=3,
        positions_count=2,
    )

    def _unexpected_snapshot(*, service_name: str):
        raise AssertionError(f"unexpected full snapshot build for {service_name}")

    monkeypatch.setattr(control_plane_mod, "build_control_plane_snapshot", _unexpected_snapshot)
    app = create_app()
    client = app.test_client()

    try:
        response = _get(client, "/operator/control-plane/open-orders")
    finally:
        runtime_state.reset_broker_status()

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["section"] == "open-orders"
    assert payload["data"]["source"] == "runtime_state"
    assert payload["data"]["available"] is True
    assert payload["data"]["open_orders_count"] == 3
    assert payload["data"]["positions_count"] == 2
    assert payload["data"]["broker_status"] == "reachable"


def test_operator_manual_overrides_post_and_delete(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    _configure_operator_auth(monkeypatch)
    monkeypatch.setenv("AI_TRADING_OPERATOR_OVERRIDE_OPERATORS", "ops@example.com")
    toggle_path = tmp_path / "runtime" / "policy_runtime_toggles.json"
    monkeypatch.setenv("AI_TRADING_POLICY_RUNTIME_TOGGLES_PATH", str(toggle_path))
    app = create_app()
    client = app.test_client()

    response = _post_json(
        client,
        "/operator/control-plane/manual-overrides",
        {
            "disabled_slices": ["ranker:bandit", "gate:max_loss"],
            "diagnostics": {"operator": "ops@example.com"},
            "source_updated_at": "2026-04-17T00:00:00Z",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    state = payload["manual_overrides"]["state"]
    assert state["disabled_slices"] == ["GATE:MAX_LOSS", "RANKER:BANDIT"]
    assert state["toggles"]["rankers"]["bandit_enabled"] is False
    assert state["toggles"]["disabled_gate_roots"] == ["MAX_LOSS"]
    persisted = json.loads(toggle_path.read_text(encoding="utf-8"))
    assert persisted["diagnostics"]["operator_id"] == "ops@example.com"

    delete = _delete(client, "/operator/control-plane/manual-overrides")
    assert delete.status_code == 200
    cleared = delete.get_json()
    assert cleared["ok"] is True
    assert cleared["manual_overrides"]["state"]["disabled_slices"] == []


def test_operator_manual_overrides_requires_auth(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    _configure_operator_auth(monkeypatch)
    monkeypatch.setenv("AI_TRADING_POLICY_RUNTIME_TOGGLES_PATH", str(tmp_path / "toggles.json"))
    app = create_app()
    client = app.test_client()

    response = client.post(
        "/operator/control-plane/manual-overrides",
        json={"disabled_slices": ["gate:max_loss"]},
    )

    assert response.status_code == 401
    payload = response.get_json()
    assert payload["ok"] is False


def test_operator_manual_overrides_requires_allowlisted_bound_operator(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    _configure_operator_auth(monkeypatch)
    monkeypatch.setenv("AI_TRADING_POLICY_RUNTIME_TOGGLES_PATH", str(tmp_path / "toggles.json"))
    app = create_app()
    client = app.test_client()

    response = _post_json(
        client,
        "/operator/control-plane/manual-overrides",
        {"disabled_slices": ["gate:max_loss"]},
    )

    assert response.status_code == 503
    payload = response.get_json()
    assert payload["ok"] is False


def test_operator_manual_overrides_rejects_malformed_body(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    _configure_operator_auth(monkeypatch)
    monkeypatch.setenv("AI_TRADING_OPERATOR_OVERRIDE_OPERATORS", "ops@example.com")
    monkeypatch.setenv("AI_TRADING_POLICY_RUNTIME_TOGGLES_PATH", str(tmp_path / "toggles.json"))
    app = create_app()
    client = app.test_client()

    response = _post_json(client, "/operator/control-plane/manual-overrides", {"diagnostics": {}})

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["ok"] is False


def test_operator_read_endpoints_require_auth(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    _configure_operator_auth(monkeypatch)
    app = create_app()
    client = app.test_client()

    response = client.get("/operator/control-plane")

    assert response.status_code == 401
    payload = response.get_json()
    assert payload["ok"] is False


def test_operator_presets_requires_auth_without_blueprint_bypass(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    _configure_operator_auth(monkeypatch)
    app = create_app()
    client = app.test_client()

    response = client.get("/operator/presets")

    assert response.status_code == 401
    payload = response.get_json()
    assert payload["ok"] is False


def test_operator_plan_post_is_not_publicly_bound(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    _configure_operator_auth(monkeypatch)
    app = create_app()
    client = app.test_client()

    routes = getattr(app, "_routes", None)
    if isinstance(routes, dict):
        assert ("/operator/plan", "POST") not in routes
        return

    response = client.post("/operator/plan", json={"preset": "balanced"})

    assert response.status_code == 405


def test_diag_requires_auth_and_returns_snapshot_when_authenticated(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    _configure_operator_auth(monkeypatch)
    app = create_app()
    app.config["broker_snapshot_fn"] = lambda: {"status": "reachable"}
    client = app.test_client()

    unauthorized = client.get("/diag")
    assert unauthorized.status_code == 401
    unauthorized_payload = unauthorized.get_json()
    assert unauthorized_payload["ok"] is False

    authorized = _get(client, "/diag")
    assert authorized.status_code == 200
    payload = authorized.get_json()
    assert "alpaca" in payload
    assert payload["broker"]["status"] == "reachable"


def test_operator_token_binding_rejects_impersonation(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setenv(
        "AI_TRADING_OPERATOR_TOKEN_MAP",
        json.dumps({"approver@example.com": "approver-token"}),
    )
    monkeypatch.setenv("AI_TRADING_OPERATOR_OVERRIDE_OPERATORS", "ops@example.com")
    monkeypatch.setenv("AI_TRADING_POLICY_RUNTIME_TOGGLES_PATH", str(tmp_path / "toggles.json"))
    app = create_app()
    client = app.test_client()

    response = client.post(
        "/operator/control-plane/manual-overrides",
        json={"disabled_slices": ["gate:max_loss"]},
        headers=_operator_headers(operator_id="ops@example.com", token="approver-token"),
    )

    assert response.status_code == 403
    payload = response.get_json()
    assert payload["ok"] is False


def test_operator_plan_builder_accepts_valid_override():
    plan = build_plan("balanced", {"max_positions": 8, "capital_cap": 0.05})

    assert plan["name"] == "balanced"
    assert plan["max_positions"] == 8
    assert plan["capital_cap"] == 0.05


def test_operator_plan_builder_rejects_invalid_override():
    try:
        build_plan("balanced", {"capital_cap": 0.9})
    except PresetValidationError as exc:
        assert "capital_cap must be between" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected PresetValidationError")


def test_operator_governance_snapshot_endpoint(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    _configure_operator_auth(monkeypatch)
    app = create_app()
    client = app.test_client()

    response = _get(client, "/operator/governance")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert isinstance(payload["governance"], dict)


def test_operator_governance_approval_endpoint(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    _configure_operator_auth(monkeypatch)
    monkeypatch.setenv("AI_TRADING_OPERATOR_APPROVERS", "ops@example.com")
    calls: dict[str, object] = {}

    class _FakePromotion:
        def __init__(self, *args, **kwargs):
            calls["base_path"] = kwargs.get("base_path")

        def record_promotion_approval(self, **kwargs):
            calls["approval"] = dict(kwargs)
            return "/tmp/promotion_approvals.jsonl"

        def list_recent_promotion_approvals(self, *, limit: int = 20):
            calls["approval_limit"] = limit
            return [{"approval_id": "approval-1", "decision": "approved"}]

    monkeypatch.setattr(promotion_mod, "ModelPromotion", _FakePromotion)
    app = create_app()
    client = app.test_client()
    response = _post_json(
        client,
        "/operator/governance/approval",
        {
            "strategy": "momentum",
            "model_id": "model-1",
            "decision": "approved",
            "note": "weekly review",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["path"] == "/tmp/promotion_approvals.jsonl"
    assert payload["approvals"][0]["approval_id"] == "approval-1"
    recorded = calls["approval"]
    assert isinstance(recorded, dict)
    assert recorded["strategy"] == "momentum"
    assert recorded["model_id"] == "model-1"
    assert recorded["approver"] == "ops@example.com"


def test_operator_governance_approval_requires_allowlisted_operator(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    _configure_operator_auth(monkeypatch)
    monkeypatch.setenv("AI_TRADING_OPERATOR_APPROVERS", "approver@example.com")
    app = create_app()
    client = app.test_client()

    response = client.post(
        "/operator/governance/approval",
        json={"strategy": "momentum", "model_id": "model-1"},
        headers=_operator_headers(operator_id="ops@example.com"),
    )

    assert response.status_code == 403
    payload = response.get_json()
    assert payload["ok"] is False


def test_operator_governance_rollback_endpoint(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    _configure_operator_auth(monkeypatch)
    monkeypatch.setenv("AI_TRADING_OPERATOR_ROLLBACK_OPERATORS", "ops@example.com")

    class _FakePromotion:
        def __init__(self, *args, **kwargs):
            return

        def rollback_to_previous_production(self, **kwargs):
            assert kwargs["strategy"] == "momentum"
            return True

        def list_recent_rollback_audit(self, *, limit: int = 20):
            assert limit == 5
            return [{"status": "rolled_back"}]

    monkeypatch.setattr(promotion_mod, "ModelPromotion", _FakePromotion)
    app = create_app()
    client = app.test_client()
    response = _post_json(
        client,
        "/operator/governance/rollback",
        {"strategy": "momentum", "reason": "manual"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["rolled_back"] is True
    assert payload["audit"][0]["status"] == "rolled_back"
