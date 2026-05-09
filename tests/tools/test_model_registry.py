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
