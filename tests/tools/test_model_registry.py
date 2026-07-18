from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ai_trading.tools import model_registry


def _metrics(edge: float, generated_at: str = "2026-05-05T20:00:00Z") -> dict[str, object]:
    return {
        "generated_at": generated_at,
        "net_edge_bps": edge,
        "sample_count": 120,
        "status": "passed",
    }


def test_register_champion_requires_manual_approval(tmp_path: Path) -> None:
    model = tmp_path / "champion.joblib"
    model.write_text("champion", encoding="utf-8")

    report = model_registry.build_model_registration(
        model_id="champion-v1",
        role="champion",
        model_path=model,
        metrics=_metrics(2.0),
        generated_at=datetime(2026, 5, 5, 21, 0, tzinfo=UTC),
    )

    assert report["status"] == "blocked"
    assert "manual_approval_required" in report["blocked_reasons"]
    assert report["promotion_authority"] is False
    assert report["registered_model"]["manual_approval"]["required_for_champion"] is True


def test_register_champion_with_approval_records_rollback_metadata(tmp_path: Path) -> None:
    old_model = tmp_path / "old.joblib"
    new_model = tmp_path / "new.joblib"
    old_model.write_text("old", encoding="utf-8")
    new_model.write_text("new", encoding="utf-8")
    generated = datetime(2026, 5, 5, 21, 0, tzinfo=UTC)
    previous = model_registry.build_model_registration(
        model_id="champion-v1",
        role="champion",
        model_path=old_model,
        metrics=_metrics(1.0),
        generated_at=generated,
        manual_approval_id="approval-1",
    )

    report = model_registry.build_model_registration(
        model_id="champion-v2",
        role="champion",
        model_path=new_model,
        metrics=_metrics(3.0),
        previous_registry=previous,
        generated_at=generated,
        manual_approval_id="approval-2",
        rollback_command="AI_TRADING_MODEL_PATH=/models/champion-v1.joblib",
    )

    assert report["status"] == "registered"
    assert report["champion"]["model_id"] == "champion-v2"
    assert report["rollback"]["previous_champion_model_id"] == "champion-v1"
    assert report["rollback"]["command"] == "AI_TRADING_MODEL_PATH=/models/champion-v1.joblib"


def test_register_challenger_blocks_stale_evidence(tmp_path: Path) -> None:
    model = tmp_path / "challenger.joblib"
    model.write_text("challenger", encoding="utf-8")

    report = model_registry.build_model_registration(
        model_id="challenger-v1",
        role="challenger",
        model_path=model,
        metrics=_metrics(4.0, generated_at="2026-05-01T20:00:00Z"),
        generated_at=datetime(2026, 5, 5, 21, 0, tzinfo=UTC),
        max_evidence_age_hours=24.0,
    )

    assert report["status"] == "blocked"
    assert "evidence_stale" in report["blocked_reasons"]
    assert report["registered_model"]["evidence_freshness"]["ok"] is False


def test_register_challenger_records_huggingface_external_metadata_research_only(
    tmp_path: Path,
) -> None:
    model = tmp_path / "challenger.joblib"
    model.write_text("challenger", encoding="utf-8")

    report = model_registry.build_model_registration(
        model_id="challenger-hf-v1",
        role="challenger",
        model_path=model,
        metrics=_metrics(2.5),
        generated_at=datetime(2026, 5, 5, 21, 0, tzinfo=UTC),
        external_metadata={
            "external_source": "huggingface",
            "external_repo_id": "open/finance-timeseries",
            "external_revision": "abc123",
            "external_license": "apache-2.0",
            "external_card_hash": "cardhash",
        },
    )

    external = report["registered_model"]["external_metadata"]
    assert external["external_source"] == "huggingface"
    assert external["local_validation_required"] is True
    assert external["runtime_authority"] is False
    assert external["promotion_authority"] is False
    assert external["live_money_authority"] is False


def test_evaluate_challenger_is_advisory_and_requires_manual_promotion(
    tmp_path: Path,
) -> None:
    champion_model = tmp_path / "champion.joblib"
    challenger_model = tmp_path / "challenger.joblib"
    champion_model.write_text("champion", encoding="utf-8")
    challenger_model.write_text("challenger", encoding="utf-8")
    generated = datetime(2026, 5, 5, 21, 0, tzinfo=UTC)
    registry = model_registry.build_model_registration(
        model_id="champion-v1",
        role="champion",
        model_path=champion_model,
        metrics=_metrics(2.0),
        generated_at=generated,
        manual_approval_id="approval-1",
    )
    registry = model_registry.build_model_registration(
        model_id="challenger-v1",
        role="challenger",
        model_path=challenger_model,
        metrics=_metrics(3.5),
        previous_registry=registry,
        generated_at=generated,
    )

    evaluation = model_registry.build_model_evaluation(
        registry=registry,
        challenger_id="challenger-v1",
        primary_metric="net_edge_bps",
        min_delta=1.0,
        generated_at=generated,
    )

    assert evaluation["status"] == "evaluated"
    assert evaluation["metrics"]["beats_champion"] is True
    assert evaluation["recommendation"] == "manual_review_for_promotion"
    assert evaluation["promotion_authority"] is False
    assert evaluation["manual_promotion_required"] is True


def test_evaluate_governed_keyed_registry_discovers_viable_identities_read_only(
    tmp_path: Path,
) -> None:
    champion_model = tmp_path / "champion.joblib"
    challenger_model = tmp_path / "challenger.joblib"
    champion_model.write_text("champion", encoding="utf-8")
    challenger_model.write_text("challenger", encoding="utf-8")
    registry = {
        "champion-live": {
            "strategy": "ml_edge",
            "registered_at": "2026-05-04T20:00:00Z",
            "artifact_path": str(champion_model),
            "active": True,
            "governance": {"status": "production", "metrics": _metrics(2.0)},
        },
        "champion-dead-newer": {
            "strategy": "ml_edge",
            "registered_at": "2026-05-05T20:00:00Z",
            "artifact_path": str(tmp_path / "dead-champion.joblib"),
            "active": True,
            "governance": {"status": "production", "metrics": _metrics(9.0)},
        },
        "challenger-shadow": {
            "strategy": "ml_edge",
            "registered_at": "2026-05-05T20:30:00Z",
            "artifact_path": str(challenger_model),
            "active": True,
            "governance": {"status": "shadow", "metrics": _metrics(3.5)},
        },
    }
    original = json.loads(json.dumps(registry))

    evaluation = model_registry.build_model_evaluation(
        registry=registry,
        primary_metric="net_edge_bps",
        min_delta=1.0,
        generated_at=datetime(2026, 5, 5, 21, 0, tzinfo=UTC),
        trusted_artifact_roots=[tmp_path],
    )

    assert evaluation["status"] == "evaluated"
    assert evaluation["registry_schema"] == "governed_keyed_mapping"
    assert evaluation["active_champion"]["model_id"] == "champion-live"
    assert evaluation["active_challenger"]["model_id"] == "challenger-shadow"
    assert evaluation["identity_discovery"]["champion"]["artifact_rejected_count"] == 1
    assert evaluation["metrics"]["champion"] == 2.0
    assert evaluation["metrics"]["challenger"] == 3.5
    assert registry == original


def test_evaluate_governed_registry_reports_shadow_when_champion_artifact_missing(
    tmp_path: Path,
) -> None:
    challenger_model = tmp_path / "challenger.joblib"
    challenger_model.write_text("challenger", encoding="utf-8")
    registry = {
        "champion-dead": {
            "strategy": "ml_edge",
            "registered_at": "2026-05-05T19:00:00Z",
            "artifact_path": str(tmp_path / "missing.joblib"),
            "governance": {"status": "production", "metrics": _metrics(2.0)},
        },
        "challenger-shadow": {
            "strategy": "ml_edge",
            "registered_at": "2026-05-05T20:00:00Z",
            "artifact_path": str(challenger_model),
            "governance": {"status": "shadow", "metrics": _metrics(3.0)},
        },
    }

    evaluation = model_registry.build_model_evaluation(
        registry=registry,
        generated_at=datetime(2026, 5, 5, 21, 0, tzinfo=UTC),
        trusted_artifact_roots=[tmp_path],
    )

    assert evaluation["status"] == "blocked"
    assert "champion_artifact_missing" in evaluation["blocked_reasons"]
    assert evaluation["active_champion"] is None
    assert evaluation["active_challenger"]["model_id"] == "challenger-shadow"
    assert evaluation["identity_discovery"]["champion"]["artifact_rejected_count"] == 1
    assert evaluation["recommendation"] == "blocked"


def test_evaluate_governed_registry_does_not_alias_mean_expectancy_to_net_edge(
    tmp_path: Path,
) -> None:
    champion_model = tmp_path / "champion.joblib"
    challenger_model = tmp_path / "challenger.joblib"
    champion_model.write_text("champion", encoding="utf-8")
    challenger_model.write_text("challenger", encoding="utf-8")
    registry = {
        "champion": {
            "strategy": "ml_edge",
            "registered_at": "2026-05-05T19:00:00Z",
            "artifact_path": str(champion_model),
            "governance": {
                "status": "production",
                "metrics": {
                    "generated_at": "2026-05-05T20:00:00Z",
                    "mean_expectancy_bps": 1.0,
                },
            },
        },
        "challenger": {
            "strategy": "ml_edge",
            "registered_at": "2026-05-05T20:00:00Z",
            "artifact_path": str(challenger_model),
            "governance": {
                "status": "shadow",
                "metrics": {
                    "generated_at": "2026-05-05T20:00:00Z",
                    "mean_expectancy_bps": 4.0,
                },
            },
        },
    }

    evaluation = model_registry.build_model_evaluation(
        registry=registry,
        primary_metric="net_edge_bps",
        generated_at=datetime(2026, 5, 5, 21, 0, tzinfo=UTC),
        trusted_artifact_roots=[tmp_path],
    )

    assert evaluation["status"] == "blocked"
    assert "champion_primary_metric_missing" in evaluation["blocked_reasons"]
    assert "challenger_primary_metric_missing" in evaluation["blocked_reasons"]
    assert evaluation["active_champion"]["model_id"] == "champion"
    assert evaluation["active_challenger"]["model_id"] == "challenger"
    assert evaluation["metrics"]["champion"] is None
    assert evaluation["metrics"]["challenger"] is None
    assert evaluation["metrics"]["delta"] is None


def test_governed_registry_rejects_existing_artifact_outside_trusted_roots(
    tmp_path: Path,
) -> None:
    trusted_root = tmp_path / "trusted"
    ephemeral_root = tmp_path / "pytest-of-aiuser"
    trusted_root.mkdir()
    ephemeral_root.mkdir()
    challenger_model = trusted_root / "challenger.joblib"
    ephemeral_model = ephemeral_root / "ml_latest.joblib"
    challenger_model.write_text("challenger", encoding="utf-8")
    ephemeral_model.write_text("pytest artifact", encoding="utf-8")
    registry = {
        "champion-stale": {
            "strategy": "ml_edge",
            "registered_at": "2026-05-05T19:00:00Z",
            "artifact_path": str(trusted_root / "missing.joblib"),
            "governance": {
                "status": "production",
                "metrics": _metrics(2.0),
                "runtime_promotion": {"model_path": str(ephemeral_model)},
            },
        },
        "challenger-shadow": {
            "strategy": "ml_edge",
            "registered_at": "2026-05-05T20:00:00Z",
            "artifact_path": str(challenger_model),
            "governance": {"status": "shadow", "metrics": _metrics(3.0)},
        },
    }

    evaluation = model_registry.build_model_evaluation(
        registry=registry,
        generated_at=datetime(2026, 5, 5, 21, 0, tzinfo=UTC),
        trusted_artifact_roots=[trusted_root],
    )

    assert evaluation["status"] == "blocked"
    assert "champion_artifact_untrusted" in evaluation["blocked_reasons"]
    assert evaluation["active_champion"] is None
    assert evaluation["active_challenger"]["model_id"] == "challenger-shadow"
    assert evaluation["artifact_trust"] == {
        "enforced": True,
        "trusted_roots": [str(trusted_root.resolve())],
    }
    rejection = evaluation["identity_discovery"]["champion"]["artifact_rejections"][0]
    assert rejection["reason"] == "artifact_untrusted"
    assert rejection["untrusted_paths"] == [str(ephemeral_model.resolve())]


def test_model_registry_cli_writes_dated_and_latest_outputs(tmp_path: Path) -> None:
    model = tmp_path / "challenger.joblib"
    metrics = tmp_path / "metrics.json"
    output_dir = tmp_path / "registry"
    latest = output_dir / "model_registry_latest.json"
    model.write_text("challenger", encoding="utf-8")
    metrics.write_text(json.dumps(_metrics(3.0)), encoding="utf-8")

    exit_code = model_registry.main(
        [
            "register",
            "--model-id",
            "challenger-v1",
            "--role",
            "challenger",
            "--model-path",
            str(model),
            "--metrics-json",
            str(metrics),
            "--generated-at",
            "2026-05-05T21:00:00Z",
            "--output-dir",
            str(output_dir),
        ]
    )

    dated = sorted(output_dir.glob("model_registry_*.json"))
    assert exit_code == 0
    assert latest.is_file()
    assert dated
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "model_registry"
    assert payload["registered_model"]["model_id"] == "challenger-v1"
    assert payload["paths"]["latest"] == str(latest)


def test_model_registry_cli_evaluate_returns_blocked_for_stale_challenger(
    tmp_path: Path,
) -> None:
    champion_model = tmp_path / "champion.joblib"
    challenger_model = tmp_path / "challenger.joblib"
    registry_path = tmp_path / "registry.json"
    output = tmp_path / "evaluation.json"
    latest = tmp_path / "evaluation_latest.json"
    champion_model.write_text("champion", encoding="utf-8")
    challenger_model.write_text("challenger", encoding="utf-8")
    generated = datetime(2026, 5, 5, 21, 0, tzinfo=UTC)
    registry = model_registry.build_model_registration(
        model_id="champion-v1",
        role="champion",
        model_path=champion_model,
        metrics=_metrics(2.0),
        generated_at=generated,
        manual_approval_id="approval-1",
    )
    registry = model_registry.build_model_registration(
        model_id="challenger-v1",
        role="challenger",
        model_path=challenger_model,
        metrics=_metrics(4.0, generated_at="2026-05-01T20:00:00Z"),
        previous_registry=registry,
        generated_at=generated,
        max_evidence_age_hours=120.0,
    )
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    exit_code = model_registry.main(
        [
            "evaluate",
            "--registry-json",
            str(registry_path),
            "--challenger-id",
            "challenger-v1",
            "--max-evidence-age-hours",
            "24",
            "--generated-at",
            "2026-05-05T21:00:00Z",
            "--output-json",
            str(output),
            "--latest-json",
            str(latest),
        ]
    )

    assert exit_code == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "challenger_evidence_stale" in payload["blocked_reasons"]
    assert latest.is_file()
