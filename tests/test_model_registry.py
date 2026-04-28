"""
Tests for model registry functionality.
"""

import json
import tempfile
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import joblib
import pytest

from ai_trading.model_registry import ModelRegistry
from ai_trading.models.artifacts import write_artifact_manifest


class TestModelRegistry:
    """Test model registry round-trip functionality."""

    def test_registry_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)
            model = {"type": "test_model", "params": {"param1": 1.0}}
            strategy = "test_strategy"
            model_type = "test_type"
            metadata = {"created_by": "test", "version": "1.0"}
            dataset_fingerprint = "test_fingerprint_123"
            tags = ["test", "unit_test"]

            model_id = registry.register_model(
                model=model,
                strategy=strategy,
                model_type=model_type,
                metadata=metadata,
                dataset_fingerprint=dataset_fingerprint,
                tags=tags,
            )

            assert isinstance(model_id, str)
            assert registry.latest_for(strategy, model_type) == model_id

            loaded_model, loaded_metadata = registry.load_model(model_id)
            assert loaded_model == model
            assert loaded_metadata["strategy"] == strategy
            assert loaded_metadata["model_type"] == model_type
            assert loaded_metadata["dataset_fingerprint"] == dataset_fingerprint
            assert loaded_metadata["tags"] == tags
            assert loaded_metadata["created_by"] == "test"
            assert loaded_metadata["version"] == "1.0"
            assert loaded_metadata["artifact_format"] == "json"

    def test_index_file_creation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)
            model_id = registry.register_model(
                model={"test": "data"},
                strategy="test_strategy",
                model_type="test_type",
            )

            index_file = Path(temp_dir) / "registry_index.json"
            assert index_file.exists()

            index_data = json.loads(index_file.read_text())
            assert model_id in index_data
            assert index_data[model_id]["strategy"] == "test_strategy"

    def test_dataset_fingerprint_verification(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)
            model_id = registry.register_model(
                model={"test": "data"},
                strategy="test_strategy",
                model_type="test_type",
                dataset_fingerprint="correct_fingerprint",
            )

            loaded_model, _loaded_metadata = registry.load_model(
                model_id,
                verify_dataset_hash=True,
                expected_dataset_fingerprint="correct_fingerprint",
            )
            assert loaded_model == {"test": "data"}

            with pytest.raises(ValueError, match="Dataset fingerprint mismatch"):
                registry.load_model(
                    model_id,
                    verify_dataset_hash=True,
                    expected_dataset_fingerprint="wrong_fingerprint",
                )

    def test_latest_for_empty_registry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)
            assert registry.latest_for("nonexistent_strategy", "nonexistent_type") is None

    def test_load_nonexistent_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)
            with pytest.raises(FileNotFoundError, match="Model nonexistent_id not found in registry"):
                registry.load_model("nonexistent_id")

    def test_model_not_json_safe_requires_artifact_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)
            with pytest.raises(RuntimeError, match="no longer serializes arbitrary Python objects"):
                registry.register_model(
                    model=object(),
                    strategy="test_strategy",
                    model_type="test_type",
                )

    def test_external_artifact_registration_loads_metadata_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)
            external_artifact = Path(temp_dir) / "runtime" / "ml_latest.joblib"
            external_artifact.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"weights": [1, 2, 3]}, external_artifact)
            manifest_path = write_artifact_manifest(
                model_path=external_artifact,
                model_version="external-v1",
            )

            model_id = registry.register_model(
                model={
                    "artifact_path": str(external_artifact),
                    "manifest_path": str(manifest_path),
                    "artifact_kind": "joblib",
                },
                strategy="ml_edge",
                model_type="joblib",
                metadata={
                    "model_path": str(external_artifact),
                    "manifest_path": str(manifest_path),
                    "version": 7,
                },
            )

            loaded_model, loaded_metadata = registry.load_model(model_id)
            assert loaded_model == {"weights": [1, 2, 3]}
            assert loaded_metadata["artifact_format"] == "external_path"
            assert loaded_metadata["artifact_path"] == str(external_artifact)
            assert loaded_metadata["manifest_path"] == str(manifest_path)
            assert loaded_metadata["version"] == 7

            external_artifact.write_bytes(b"tampered")
            with pytest.raises(RuntimeError, match="CHECKSUM_MISMATCH"):
                registry.load_model(model_id)

    def test_metadata_class_path_conversion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)
            model_id = registry.register_model(
                model={"a": 1},
                strategy="strat",
                model_type="dict",
                metadata={"cls": Mock},
            )
            _model, meta = registry.load_model(model_id)
            assert meta["cls"] == "unittest.mock.Mock"

    def test_list_models_empty_registry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)
            assert registry.list_models() == []

    def test_list_models_populated_registry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)
            m1 = registry.register_model({"a": 1}, "s1", "t1")
            m2 = registry.register_model({"b": 2}, "s2", "t2")
            model_ids = cast(list[str], registry.list_models())
            assert sorted(model_ids) == sorted([m1, m2])

    def test_sequential_registration_updates_latest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)
            model = {"a": 1}

            first_id = registry.register_model(model, "s", "t", metadata={"version": 1})
            second_id = registry.register_model(model, "s", "t", metadata={"version": 2})

            assert first_id != second_id
            assert registry.latest_for("s", "t") == second_id

            _old_model, old_meta = registry.load_model(first_id)
            _new_model, new_meta = registry.load_model(second_id)
            assert old_meta["version"] == 1
            assert new_meta["version"] == 2

    def test_get_production_model_returns_newest_registration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)
            older = registry.register_model({"a": 1}, "ml_edge", "dict")
            newer = registry.register_model({"a": 2}, "ml_edge", "dict")
            registry.update_governance_status(older, "production")
            registry.update_governance_status(newer, "production")

            production = registry.get_production_model("ml_edge")

            assert production is not None
            assert production[0] == newer

    def test_viable_production_lookup_skips_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)
            viable = registry.register_model({"a": 1}, "ml_edge", "dict")
            stale = registry.register_model({"a": 2}, "ml_edge", "dict")
            registry.update_governance_status(viable, "production")
            registry.update_governance_status(stale, "production")

            stale_artifact = Path(registry.model_index[stale]["artifact_path"])
            stale_artifact.unlink()

            production = registry.get_viable_production_model("ml_edge")

            assert production is not None
            prod_id, info = production
            assert prod_id == viable
            assert info["production_path_source"] == "artifact"
            assert Path(info["production_path"]).is_file()

    def test_record_runtime_promotion_persists_runtime_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)
            model_id = registry.register_model({"a": 1}, "ml_edge", "dict")
            registry.update_governance_status(model_id, "production")
            artifact_path = Path(registry.model_index[model_id]["artifact_path"])
            runtime_model_path = Path(temp_dir) / "runtime" / "ml_latest.joblib"
            runtime_model_path.parent.mkdir(parents=True, exist_ok=True)
            runtime_model_path.write_bytes(artifact_path.read_bytes())
            artifact_path.unlink()

            registry.record_runtime_promotion(
                model_id,
                model_path=runtime_model_path,
                manifest_path=runtime_model_path.with_suffix(".manifest.json"),
            )

            refreshed = ModelRegistry(temp_dir)
            production = refreshed.get_viable_production_model("ml_edge")

            assert production is not None
            prod_id, info = production
            assert prod_id == model_id
            assert info["production_path_source"] == "runtime_promotion"
            assert info["production_path"] == str(runtime_model_path)
            runtime_promotion = refreshed.model_index[model_id]["governance"]["runtime_promotion"]
            assert runtime_promotion["model_path"] == str(runtime_model_path)
