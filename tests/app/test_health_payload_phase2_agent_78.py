from __future__ import annotations

from pathlib import Path

import ai_trading.health_payload as health_payload


def test_timestamp_age_seconds_handles_naive_future_and_bad_values(monkeypatch) -> None:
    assert health_payload._timestamp_age_seconds(None) is None
    assert health_payload._timestamp_age_seconds("") is None
    assert health_payload._timestamp_age_seconds("not-a-date") is None
    assert health_payload._timestamp_age_seconds("2999-01-01T00:00:00") == 0.0


def test_database_readiness_disabled_short_circuits(monkeypatch) -> None:
    monkeypatch.setattr(health_payload, "_env_bool", lambda name, default: False)

    assert health_payload._database_readiness_snapshot() == {"enabled": False}


def test_database_readiness_reports_unconfigured(monkeypatch) -> None:
    monkeypatch.setattr(
        health_payload,
        "_env_bool",
        lambda name, default: name == "AI_TRADING_HEALTH_DB_READINESS_ENABLED",
    )
    monkeypatch.setattr(
        health_payload,
        "get_env",
        lambda name, default="", **_kwargs: "" if name != "AI_TRADING_OMS_EXPECTED_ALEMBIC_REVISION" else default,
    )

    payload = health_payload._database_readiness_snapshot()

    assert payload == {
        "enabled": True,
        "configured": False,
        "ok": True,
        "reason": "database_not_configured",
    }


def test_database_readiness_fails_unconfigured_when_required(monkeypatch) -> None:
    monkeypatch.setattr(health_payload, "_env_bool", lambda name, default: True)
    monkeypatch.setattr(
        health_payload,
        "get_env",
        lambda name, default="", **_kwargs: "" if name != "AI_TRADING_OMS_EXPECTED_ALEMBIC_REVISION" else default,
    )

    payload = health_payload._database_readiness_snapshot()

    assert payload["configured"] is False
    assert payload["ok"] is False


def test_read_json_mapping_artifact_ignores_missing_invalid_and_non_mapping(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        health_payload,
        "resolve_runtime_artifact_path",
        lambda configured_path, *, default_relative: Path(configured_path),
    )

    missing, missing_path = health_payload._read_json_mapping_artifact(
        configured_path=str(tmp_path / "missing.json"),
        default_relative="runtime/missing.json",
    )
    assert missing == {}
    assert missing_path == tmp_path / "missing.json"

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    payload, _ = health_payload._read_json_mapping_artifact(
        configured_path=str(invalid),
        default_relative="runtime/invalid.json",
    )
    assert payload == {}

    list_json = tmp_path / "list.json"
    list_json.write_text("[1, 2]", encoding="utf-8")
    payload, _ = health_payload._read_json_mapping_artifact(
        configured_path=str(list_json),
        default_relative="runtime/list.json",
    )
    assert payload == {}


def test_build_alpaca_health_payload_enrich_failure_preserves_context(monkeypatch) -> None:
    payload = health_payload.build_alpaca_health_payload(
        {"has_key": "yes", "base_url": "https://paper-api.example.test"},
        enrich_from_runtime_env=False,
    )

    assert payload["has_key"] is True
    assert payload["has_secret"] is False
    assert payload["paper"] is True


def test_canonical_healthz_error_and_extras_override(monkeypatch) -> None:
    monkeypatch.setattr(
        health_payload,
        "build_service_health_payload",
        lambda **_kwargs: {"ok": True, "status": "healthy", "service": "svc"},
    )

    payload = health_payload.build_canonical_healthz_payload(
        service_name="svc",
        error="database unavailable",
        extras={"build": "abc123"},
    )

    assert payload["ok"] is False
    assert payload["status"] == "degraded"
    assert payload["error"] == "database unavailable"
    assert payload["build"] == "abc123"
