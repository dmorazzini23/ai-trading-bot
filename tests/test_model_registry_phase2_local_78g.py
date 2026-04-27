from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ai_trading import model_registry as mr


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
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"weights": [1, 2, 3]}')
    registry = mr.ModelRegistry(tmp_path / "registry")

    model_id = registry.register_model(
        {"paths": {"model_path": str(artifact)}},
        "mean reversion!",
        "json model",
        metadata={
            "created": datetime(2024, 1, 2, tzinfo=UTC),
            "path": artifact,
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

    assert loaded is None
    assert metadata["artifact_format"] == "external_path"
    assert metadata["path"] == str(artifact)
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
