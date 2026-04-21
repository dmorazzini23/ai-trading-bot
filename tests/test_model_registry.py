"""
Tests for model registry functionality.
"""
import json
import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import pytest
from ai_trading.model_registry import ModelRegistry
from ai_trading.utils import safe_pickle


class TestModelRegistry:
    """Test model registry round-trip functionality."""

    def test_registry_roundtrip(self):
        """Test register, latest_for, and load_model cycle."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create registry in temp directory
            registry = ModelRegistry(temp_dir)

            # Create a simple test model (using a dict as a trivial model)
            model = {"type": "test_model", "params": {"param1": 1.0}}
            strategy = "test_strategy"
            model_type = "test_type"
            metadata = {"created_by": "test", "version": "1.0"}
            dataset_fingerprint = "test_fingerprint_123"
            tags = ["test", "unit_test"]

            # Register the model
            model_id = registry.register_model(
                model=model,
                strategy=strategy,
                model_type=model_type,
                metadata=metadata,
                dataset_fingerprint=dataset_fingerprint,
                tags=tags
            )

            # Verify model ID is returned
            assert isinstance(model_id, str)
            assert len(model_id) > 0

            # Test latest_for returns the correct ID
            latest_id = registry.latest_for(strategy, model_type)
            assert latest_id == model_id

            # Test load_model returns correct data
            loaded_model, loaded_metadata = registry.load_model(model_id)
            assert loaded_model == model
            assert loaded_metadata["strategy"] == strategy
            assert loaded_metadata["model_type"] == model_type
            assert loaded_metadata["dataset_fingerprint"] == dataset_fingerprint
            assert loaded_metadata["tags"] == tags

            # Verify metadata was merged correctly
            assert loaded_metadata["created_by"] == "test"
            assert loaded_metadata["version"] == "1.0"

    def test_index_file_creation(self):
        """Test that index file is created and maintained."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            # Register a model
            model = {"test": "data"}
            model_id = registry.register_model(
                model=model,
                strategy="test_strategy",
                model_type="test_type"
            )

            # Check that index file exists
            index_file = Path(temp_dir) / "registry_index.json"
            assert index_file.exists()

            # Verify index contains our model
            with open(index_file) as f:
                index_data = json.load(f)
            assert model_id in index_data
            assert index_data[model_id]["strategy"] == "test_strategy"

    def test_dataset_fingerprint_verification(self):
        """Test dataset fingerprint mismatch raises ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            # Register model with fingerprint
            model = {"test": "data"}
            model_id = registry.register_model(
                model=model,
                strategy="test_strategy",
                model_type="test_type",
                dataset_fingerprint="correct_fingerprint"
            )

            # Loading with correct fingerprint should work
            loaded_model, loaded_metadata = registry.load_model(
                model_id,
                verify_dataset_hash=True,
                expected_dataset_fingerprint="correct_fingerprint"
            )
            assert loaded_model == model

            # Loading with wrong fingerprint should raise ValueError
            with pytest.raises(ValueError, match="Dataset fingerprint mismatch"):
                registry.load_model(
                    model_id,
                    verify_dataset_hash=True,
                    expected_dataset_fingerprint="wrong_fingerprint"
                )

    def test_latest_for_empty_registry(self):
        """Test latest_for returns None for non-existent models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            # Empty registry should return None
            latest_id = registry.latest_for("nonexistent_strategy", "nonexistent_type")
            assert latest_id is None

    def test_load_nonexistent_model(self):
        """Test loading non-existent model raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            with pytest.raises(FileNotFoundError, match="Model nonexistent_id not found in registry"):
                registry.load_model("nonexistent_id")

    def test_model_not_picklable(self):
        """Test that non-picklable models raise RuntimeError when all picklers fail."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)

            model = object()

            def fail(*_a, **_k):  # pragma: no cover - trivial
                raise Exception("Cannot pickle this object")

            failing_picklers = [
                SimpleNamespace(name="primary", dumps=fail, loads=lambda data: data),
                SimpleNamespace(name="fallback", dumps=fail, loads=lambda data: data),
            ]

            with pytest.raises(RuntimeError, match="Model not picklable"):
                with patch.object(
                    ModelRegistry,
                    "_available_picklers",
                    return_value=failing_picklers,
                ):
                    registry.register_model(
                        model=model,
                        strategy="test_strategy",
                        model_type="test_type",
                    )

    def test_pickle_fallback(self):
        """pickle failure should fall back to cloudpickle or dill."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)
            model = {"a": 1}

            picklers = [
                SimpleNamespace(
                    name="primary",
                    dumps=lambda _obj: (_ for _ in ()).throw(Exception("boom")),
                    loads=lambda data: data,
                ),
                SimpleNamespace(name="fallback", dumps=pickle.dumps, loads=pickle.loads),
            ]

            with patch.object(ModelRegistry, "_available_picklers", return_value=picklers):
                model_id = registry.register_model(model, "s", "t")
            assert model_id in registry.model_index
            assert registry.model_index[model_id]["pickler"] == "fallback"

    def test_load_model_blocks_unsafe_deserialization_outside_test_runtime(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)
            model_id = registry.register_model({"a": 1}, "s", "t")
            monkeypatch.setattr(safe_pickle, "is_test_runtime", lambda: False)
            monkeypatch.delenv("AI_TRADING_ALLOW_UNSAFE_MODEL_DESERIALIZATION", raising=False)

            with pytest.raises(RuntimeError, match="unsafe generic model deserialization"):
                registry.load_model(model_id)

    def test_metadata_class_path_conversion(self):
        """Metadata containing classes should be stored as dotted path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)
            model = {"a": 1}
            model_id = registry.register_model(
                model=model,
                strategy="strat",
                model_type="dict",
                metadata={"cls": Mock},
            )
            _model, meta = registry.load_model(model_id)
            assert meta["cls"] == "unittest.mock.Mock"

    def test_list_models_empty_registry(self):
        """list_models returns an empty list when no models registered."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)
            assert registry.list_models() == []

    def test_list_models_populated_registry(self):
        """list_models returns all registered model IDs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)
            m1 = registry.register_model({"a": 1}, "s1", "t1")
            m2 = registry.register_model({"b": 2}, "s2", "t2")
            model_ids = cast(list[str], registry.list_models())
            assert sorted(model_ids) == sorted([m1, m2])

    def test_sequential_registration_updates_latest(self):
        """Sequential registrations of identical payloads should succeed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)
            model = {"a": 1}

            first_id = registry.register_model(
                model,
                "s",
                "t",
                metadata={"version": 1},
            )
            second_id = registry.register_model(
                model,
                "s",
                "t",
                metadata={"version": 2},
            )

            assert first_id != second_id

            latest_id = registry.latest_for("s", "t")
            assert latest_id == second_id

            _old_model, old_meta = registry.load_model(first_id)
            assert old_meta["version"] == 1

            _new_model, new_meta = registry.load_model(second_id)
            assert new_meta["version"] == 2

    def test_get_production_model_returns_newest_registration(self):
        """Production lookup should prefer the newest registered production model."""
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
        """Viable production lookup should ignore stale production entries."""
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
        """Runtime promotion metadata should be durable and usable for viable lookup."""
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
