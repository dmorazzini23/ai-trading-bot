from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pytest

from ai_trading import model_registry as mr
from ai_trading.models.artifacts import write_artifact_manifest


def test_legacy_registry_helpers_handle_activation_and_malformed_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry_path = tmp_path / "registry.json"
    models_dir = tmp_path / "models"
    eval_dir = tmp_path / "eval"
    monkeypatch.setattr(mr, "MODELS_DIR", models_dir)
    monkeypatch.setattr(mr, "_REGISTRY_PATH", registry_path)
    monkeypatch.setattr(mr, "_EVAL_DIR", eval_dir)

    registry_path.write_text(json.dumps(["not", "a", "dict"]))
    assert mr._load_registry() == {}

    mr.register_model("AAPL", "v1", tmp_path / "model-v1.json", {"score": 1}, activate=False)
    assert mr.get_active_model_meta("AAPL") is None

    mr.set_active_model("AAPL", "missing")
    assert mr.get_active_model_meta("AAPL") is None

    mr.set_active_model("AAPL", "v1")
    active = mr.get_active_model_meta("AAPL")
    assert active is not None
    assert active["meta"] == {"score": 1}

    payload = json.loads(registry_path.read_text())
    payload["MSFT"] = {"active": "v1", "versions": []}
    payload["TSLA"] = {"active": "v1", "versions": {"v1": []}}
    registry_path.write_text(json.dumps(payload))

    assert mr.get_active_model_meta("missing") is None
    assert mr.get_active_model_meta("MSFT") is None
    assert mr.get_active_model_meta("TSLA") is None

    mr.record_evaluation("AAPL", {"sharpe": 1.2})
    mr.record_evaluation("AAPL", {"sharpe": 1.3})

    assert [row["sharpe"] for row in mr.list_evaluations("AAPL", limit=1)] == [1.3]
    assert mr.list_evaluations("MSFT") == []

    (eval_dir / "BAD.jsonl").write_text("{not-json}\n")
    assert mr.list_evaluations("BAD") == []


def test_model_registry_external_artifacts_metadata_and_production_paths(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.joblib"
    joblib.dump({"weights": [1, 2, 3]}, artifact)
    manifest = write_artifact_manifest(model_path=artifact, model_version="phase2-local")
    registry = mr.ModelRegistry(tmp_path / "registry")

    model_id = registry.register_model(
        {"paths": {"model_path": str(artifact), "manifest_path": str(manifest)}},
        "mean reversion!",
        "json model",
        metadata={
            "created": datetime(2024, 1, 2, tzinfo=UTC),
            "path": artifact,
            "manifest_path": manifest,
            "klass": mr.ModelRegistry,
            "nested": {"when": datetime(2024, 1, 3, tzinfo=UTC)},
        },
        dataset_fingerprint="dataset-1",
        tags=["prod", None, 7],  # type: ignore[list-item]
    )

    loaded, metadata = registry.load_model(
        model_id,
        verify_dataset_hash=True,
        expected_dataset_fingerprint="dataset-1",
    )

    assert loaded == {"weights": [1, 2, 3]}
    assert metadata["artifact_format"] == "external_path"
    assert metadata["path"] == str(artifact)
    assert metadata["manifest_path"] == str(manifest)
    assert metadata["klass"] == "ai_trading.model_registry.ModelRegistry"
    assert metadata["tags"] == ["prod", "7"]
    assert registry.latest_for("mean reversion!", "json model") == model_id

    registry.update_governance_status(
        model_id,
        "production",
        {"runtime_promotion": {"model_path": artifact}},
    )
    production_id, production_info = registry.get_viable_production_model("mean reversion!")

    assert production_id == model_id
    assert production_info["production_path"] == str(artifact)
    assert production_info["production_path_source"] == "runtime_promotion"
    assert registry.get_production_model("mean reversion!")[0] == model_id


def test_model_registry_rich_edge_errors_and_filters(tmp_path: Path) -> None:
    registry = mr.ModelRegistry(tmp_path / "registry")
    inactive_id = registry.register_model({"value": 1}, "alpha", "dict", activate=False)
    active_id = registry.register_model({"value": 2}, "alpha", "dict", dataset_fingerprint="fp")
    shadow_id = registry.register_model({"value": 3}, "alpha", "dict")

    registry.update_governance_status(shadow_id, "shadow", {"note": Path("note.txt")})

    assert registry.latest_for("alpha", "dict") == shadow_id
    assert set(registry.list_models()) == {shadow_id, active_id, inactive_id}
    assert {
        entry["model_id"] for entry in registry.list_models(active_only=True)
    } == {shadow_id, active_id}
    assert registry.get_shadow_models("alpha")[0][0] == shadow_id

    with pytest.raises(ValueError, match="Dataset fingerprint missing"):
        registry.load_model(inactive_id, verify_dataset_hash=True)
    with pytest.raises(ValueError, match="Dataset fingerprint mismatch"):
        registry.load_model(active_id, verify_dataset_hash=True)
    with pytest.raises(ValueError, match="not found"):
        registry.update_governance_status("missing", "production")
    with pytest.raises(ValueError, match="runtime model path"):
        registry.record_runtime_promotion(active_id, model_path=" ")
    with pytest.raises(ValueError, match="not found"):
        registry.record_runtime_promotion("missing", model_path=tmp_path / "model.json")

    registry.model_index[active_id]["artifact_format"] = "pickle"
    with pytest.raises(RuntimeError, match="unsupported artifact format"):
        registry.load_model(active_id)

    registry.model_index[active_id]["artifact_format"] = "json"
    Path(registry.model_index[active_id]["artifact_path"]).write_text("{bad-json}")
    with pytest.raises(RuntimeError, match="Failed to load model"):
        registry.load_model(active_id)

    Path(registry.model_index[active_id]["artifact_path"]).unlink()
    with pytest.raises(FileNotFoundError, match="Artifact"):
        registry.load_model(active_id)


def test_shadow_candidate_registration_is_idempotent_and_retirement_is_safe(
    tmp_path: Path,
) -> None:
    registry = mr.ModelRegistry(tmp_path / "registry")
    scope = {
        "horizon_bars": 1,
        "label_objective": "net_markout",
        "evaluation_window": "2026-h1",
    }
    candidate_args = {
        "strategy": "replay_aligned_markout",
        "model_type": "logistic",
        "dataset_fingerprint": "dataset-1",
        "experiment_signature": "experiment-1",
        "comparable_scope": scope,
        "metrics": {"net_edge_bps": 2.0},
    }

    first_id, first_created = registry.register_shadow_candidate(
        {"weights": [1.0]},
        **candidate_args,
    )
    repeated_id, repeated_created = registry.register_shadow_candidate(
        {"weights": [1.0]},
        **candidate_args,
    )

    assert first_created is True
    assert repeated_created is False
    assert repeated_id == first_id
    assert len(registry.model_index) == 1
    governance = registry.model_index[first_id]["governance"]
    assert governance["status"] == "shadow"
    assert governance["promotion_authority"] is False
    assert governance["runtime_authority"] is False
    assert governance["live_money_authority"] is False
    assert governance["manual_production_promotion_required"] is True

    dominated_id, _ = registry.register_shadow_candidate(
        {"weights": [0.5]},
        strategy="replay_aligned_markout",
        model_type="logistic",
        dataset_fingerprint="dataset-0",
        experiment_signature="experiment-0",
        comparable_scope=scope,
        metrics={"net_edge_bps": 1.0},
    )
    equal_id, _ = registry.register_shadow_candidate(
        {"weights": [0.7]},
        strategy="replay_aligned_markout",
        model_type="logistic",
        dataset_fingerprint="dataset-equal",
        experiment_signature="experiment-equal",
        comparable_scope=scope,
        metrics={"net_edge_bps": 2.0},
    )
    other_scope_id, _ = registry.register_shadow_candidate(
        {"weights": [0.1]},
        strategy="replay_aligned_markout",
        model_type="logistic",
        dataset_fingerprint="dataset-other",
        experiment_signature="experiment-other",
        comparable_scope=scope | {"horizon_bars": 5},
        metrics={"net_edge_bps": -5.0},
    )
    production_id, _ = registry.register_shadow_candidate(
        {"weights": [-1.0]},
        strategy="replay_aligned_markout",
        model_type="logistic",
        dataset_fingerprint="dataset-production",
        experiment_signature="experiment-production",
        comparable_scope=scope,
        metrics={"net_edge_bps": -10.0},
    )
    registry.update_governance_status(production_id, "production")
    dominated_artifact = Path(registry.model_index[dominated_id]["artifact_path"])

    retired = registry.retire_dominated_shadow_candidates(
        first_id,
        primary_metric="net_edge_bps",
        stale_before=datetime.now(UTC),
    )

    assert retired == [dominated_id]
    assert registry.model_index[dominated_id]["active"] is False
    retired_governance = registry.model_index[dominated_id]["governance"]
    assert retired_governance["status"] == "retired"
    assert retired_governance["replacement_model_id"] == first_id
    assert retired_governance["artifact_retained"] is True
    assert dominated_artifact.is_file()
    assert registry.model_index[equal_id]["governance"]["status"] == "shadow"
    assert registry.model_index[other_scope_id]["governance"]["status"] == "shadow"
    assert registry.model_index[production_id]["governance"]["status"] == "production"

    reloaded = mr.ModelRegistry(tmp_path / "registry")
    assert reloaded.model_index[dominated_id]["governance"]["status"] == "retired"
    assert reloaded.model_index[dominated_id]["active"] is False


def test_candidate_identity_collision_fails_closed(tmp_path: Path) -> None:
    registry = mr.ModelRegistry(tmp_path / "registry")
    identity = "stable-candidate"
    registry.register_model(
        {"weights": [1.0]},
        "alpha",
        "json",
        dataset_fingerprint="dataset",
        candidate_identity=identity,
    )

    with pytest.raises(ValueError, match="candidate identity collision"):
        registry.register_model(
            {"weights": [2.0]},
            "alpha",
            "json",
            dataset_fingerprint="dataset",
            candidate_identity=identity,
        )
