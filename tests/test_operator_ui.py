from __future__ import annotations

import json
from types import SimpleNamespace

import ai_trading.app as app_mod
import ai_trading.governance.promotion as promotion_mod

from ai_trading.app import create_app
from ai_trading.operator_presets import PresetValidationError, build_plan


def _post_json_compat(client, path: str, payload: dict[str, object]):
    post = getattr(client, "post", None)
    if callable(post):
        return post(path, json=payload)
    compat_path = f"{path}/update"
    request_original = getattr(app_mod, "request", None)
    setattr(app_mod, "request", SimpleNamespace(get_json=lambda silent=True: payload))
    try:
        return client.get(compat_path)
    finally:
        setattr(app_mod, "request", request_original)


def _delete_compat(client, path: str):
    delete = getattr(client, "delete", None)
    if callable(delete):
        return delete(path)
    compat_path = f"{path}/clear"
    return client.get(compat_path)


def test_operator_presets_endpoint(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    app = create_app()
    client = app.test_client()

    response = client.get("/operator/presets")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    names = {item["name"] for item in payload["presets"]}
    assert {"conservative", "balanced", "aggressive"} <= names


def test_operator_plan_endpoint_returns_default_plan(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    app = create_app()
    client = app.test_client()

    response = client.get("/operator/plan")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["plan"]["name"] == "balanced"


def test_operator_control_plane_snapshot_endpoint(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    app = create_app()
    client = app.test_client()

    response = client.get("/operator/control-plane")

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
    app = create_app()
    client = app.test_client()

    response = client.get("/operator/control-plane/services")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["section"] == "services"
    assert payload["data"]["risk_approval"]["owner"] == "ai_trading.services.risk_approval"


def test_operator_manual_overrides_post_and_delete(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    toggle_path = tmp_path / "runtime" / "policy_runtime_toggles.json"
    monkeypatch.setenv("AI_TRADING_POLICY_RUNTIME_TOGGLES_PATH", str(toggle_path))
    app = create_app()
    client = app.test_client()

    response = _post_json_compat(
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
    assert persisted["diagnostics"]["operator"] == "ops@example.com"

    delete = _delete_compat(client, "/operator/control-plane/manual-overrides")
    assert delete.status_code == 200
    cleared = delete.get_json()
    assert cleared["ok"] is True
    assert cleared["manual_overrides"]["state"]["disabled_slices"] == []


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
    app = create_app()
    client = app.test_client()

    response = client.get("/operator/governance")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert isinstance(payload["governance"], dict)


def test_operator_governance_approval_endpoint(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
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
    response = _post_json_compat(
        client,
        "/operator/governance/approval",
        {
            "strategy": "momentum",
            "model_id": "model-1",
            "approver": "ops@example.com",
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


def test_operator_governance_rollback_endpoint(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")

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
    response = _post_json_compat(
        client,
        "/operator/governance/rollback",
        {"strategy": "momentum", "reason": "manual"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["rolled_back"] is True
    assert payload["audit"][0]["status"] == "rolled_back"
