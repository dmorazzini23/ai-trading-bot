from __future__ import annotations

from ai_trading.app import create_app
from ai_trading.operator_presets import PresetValidationError, build_plan


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
