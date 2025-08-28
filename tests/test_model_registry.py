"""
Tests for model registry functionality.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from ai_trading.model_registry import ModelRegistry


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

            mock_model = Mock()
            _ = pytest.importorskip("cloudpickle")
            dill_mod = pytest.importorskip("dill")

            def fail(*_a, **_k):  # pragma: no cover - trivial
                raise Exception("Cannot pickle this object")

            with pytest.raises(RuntimeError, match="Model not picklable"):
                with patch("pickle.dumps", side_effect=fail), patch(
                    "cloudpickle.dumps", side_effect=fail
                ), patch.object(dill_mod, "dumps", side_effect=fail):
                    registry.register_model(
                        model=mock_model,
                        strategy="test_strategy",
                        model_type="test_type",
                    )

    def test_pickle_fallback(self):
        """pickle failure should fall back to cloudpickle or dill."""
        _ = pytest.importorskip("cloudpickle")
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)
            model = Mock()

            with patch("pickle.dumps", side_effect=Exception("boom")):
                model_id = registry.register_model(model, "s", "t")
            assert model_id in registry.model_index

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
