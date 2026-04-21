"""Test model registry register -> latest_for -> load_model workflow."""

import tempfile

import pytest

from ai_trading.model_registry import ModelRegistry


def test_model_registry_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(base_path=tmpdir)
        model = {"weights": [0.1, 0.2, 0.3], "bias": 0.5}

        model_id = registry.register_model(
            model=model,
            strategy="test_strategy",
            model_type="json_descriptor",
            metadata={"test": "value"},
            dataset_fingerprint="abc123",
            tags=["test", "json"],
        )

        assert model_id is not None
        assert "test_strategy" in model_id
        assert "json_descriptor" in model_id

        latest_id = registry.latest_for("test_strategy", "json_descriptor")
        assert latest_id == model_id

        loaded_model, metadata = registry.load_model(model_id)
        assert loaded_model == model
        assert metadata["test"] == "value"
        assert metadata["dataset_fingerprint"] == "abc123"
        assert metadata["tags"] == ["test", "json"]
        assert metadata["artifact_format"] == "json"

        loaded_model2, _metadata2 = registry.load_model(
            model_id,
            verify_dataset_hash=True,
            expected_dataset_fingerprint="abc123",
        )
        assert loaded_model2 == model

        with pytest.raises(ValueError, match="Dataset fingerprint mismatch"):
            registry.load_model(
                model_id,
                verify_dataset_hash=True,
                expected_dataset_fingerprint="wrong_fingerprint",
            )


def test_model_registry_multiple_registrations():
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(base_path=tmpdir)

        id1 = registry.register_model({"model": 1}, "strat1", "json", metadata={"version": 1})
        id2 = registry.register_model({"model": 2}, "strat1", "json", metadata={"version": 2})

        latest = registry.latest_for("strat1", "json")
        assert latest == id2

        loaded1, meta1 = registry.load_model(id1)
        loaded2, meta2 = registry.load_model(id2)

        assert loaded1 == {"model": 1}
        assert loaded2 == {"model": 2}
        assert meta1["version"] == 1
        assert meta2["version"] == 2


def test_model_registry_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        registry1 = ModelRegistry(base_path=tmpdir)
        model_id = registry1.register_model({"persisted": True}, "test", "json")

        registry2 = ModelRegistry(base_path=tmpdir)

        latest = registry2.latest_for("test", "json")
        assert latest == model_id

        loaded_model, _metadata = registry2.load_model(model_id)
        assert loaded_model == {"persisted": True}


def test_model_registry_errors():
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(base_path=tmpdir)

        with pytest.raises(FileNotFoundError):
            registry.load_model("nonexistent-model-id")

        assert registry.latest_for("nonexistent", "strategy") is None

        with pytest.raises(RuntimeError, match="no longer serializes arbitrary Python objects"):
            registry.register_model(object(), "test", "dict")
