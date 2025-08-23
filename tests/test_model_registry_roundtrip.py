"""Test model registry register → latest_for → load_model workflow."""

import tempfile

import numpy as np
import pytest
pytest.importorskip("sklearn", reason="Optional heavy dependency; guard at import time")  # AI-AGENT-REF: guard sklearn
from ai_trading.model_registry import ModelRegistry
from sklearn.linear_model import LinearRegression


def test_model_registry_roundtrip():
    """Test complete workflow: register_model → latest_for → load_model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(base_path=tmpdir)

        # Create a simple model
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])
        model = LinearRegression()
        model.fit(X, y)

        # Register the model
        model_id = registry.register_model(
            model=model,
            strategy="test_strategy",
            model_type="linear_regression",
            metadata={"test": "value"},
            dataset_fingerprint="abc123",
            tags=["test", "linear"]
        )

        # Verify registration
        assert model_id is not None
        assert "test_strategy" in model_id
        assert "linear_regression" in model_id

        # Test latest_for
        latest_id = registry.latest_for("test_strategy", "linear_regression")
        assert latest_id == model_id

        # Test loading
        loaded_model, metadata = registry.load_model(model_id)
        assert isinstance(loaded_model, LinearRegression)
        assert metadata["test"] == "value"
        assert metadata["dataset_fingerprint"] == "abc123"
        assert metadata["tags"] == ["test", "linear"]

        # Test model functionality after loading
        prediction = loaded_model.predict([[7, 8]])
        assert prediction is not None

        # Test dataset fingerprint verification
        loaded_model2, metadata2 = registry.load_model(
            model_id,
            verify_dataset_hash=True,
            expected_dataset_fingerprint="abc123"
        )
        assert isinstance(loaded_model2, LinearRegression)

        # Test fingerprint mismatch
        with pytest.raises(ValueError, match="Dataset fingerprint mismatch"):
            registry.load_model(
                model_id,
                verify_dataset_hash=True,
                expected_dataset_fingerprint="wrong_fingerprint"
            )


def test_model_registry_multiple_registrations():
    """Test registry handles multiple models correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(base_path=tmpdir)

        # Register multiple models
        model1 = LinearRegression()
        model2 = LinearRegression()

        id1 = registry.register_model(model1, "strat1", "linear", metadata={"version": 1})
        id2 = registry.register_model(model2, "strat1", "linear", metadata={"version": 2})

        # Latest should return the most recent
        latest = registry.latest_for("strat1", "linear")
        assert latest == id2

        # Both models should be loadable
        loaded1, meta1 = registry.load_model(id1)
        loaded2, meta2 = registry.load_model(id2)

        assert meta1["version"] == 1
        assert meta2["version"] == 2


def test_model_registry_persistence():
    """Test that registry persists across instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create first registry instance
        registry1 = ModelRegistry(base_path=tmpdir)
        model = LinearRegression()
        model_id = registry1.register_model(model, "test", "linear")

        # Create second registry instance pointing to same directory
        registry2 = ModelRegistry(base_path=tmpdir)

        # Should be able to find the model
        latest = registry2.latest_for("test", "linear")
        assert latest == model_id

        loaded_model, metadata = registry2.load_model(model_id)
        assert isinstance(loaded_model, LinearRegression)


def test_model_registry_errors():
    """Test error conditions in model registry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(base_path=tmpdir)

        # Test loading non-existent model
        with pytest.raises(FileNotFoundError):
            registry.load_model("nonexistent-model-id")

        # Test latest_for with no matches
        result = registry.latest_for("nonexistent", "strategy")
        assert result is None

        # Test non-picklable model
        def non_picklable(x):
            return x  # lambda is not picklable
        with pytest.raises(RuntimeError, match="Model not picklable"):
            registry.register_model(non_picklable, "test", "lambda")
