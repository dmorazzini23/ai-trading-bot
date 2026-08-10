#!/usr/bin/env python3
"""
Test artifacts directory creation and environment variable overrides.
"""

import os
import tempfile


def test_walkforward_artifacts_directory():
    """Test that walkforward creates artifacts directory with env override."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set custom artifacts directory
        custom_artifacts = os.path.join(temp_dir, "custom_artifacts")
        os.environ["ARTIFACTS_DIR"] = custom_artifacts

        try:
            # Import and create evaluator
            from ai_trading.evaluation.walkforward import WalkForwardEvaluator

            WalkForwardEvaluator()

            # Check that directory was created
            expected_dir = os.path.join(custom_artifacts, "walkforward")
            assert os.path.exists(expected_dir), f"Directory {expected_dir} should exist"

        finally:
            # Clean up environment variable
            if "ARTIFACTS_DIR" in os.environ:
                del os.environ["ARTIFACTS_DIR"]


def test_model_registry_directory(monkeypatch):
    """Test that model registry creates directory with env override."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set custom model registry directory
        custom_registry = os.path.join(temp_dir, "custom_models")
        monkeypatch.setenv("MODEL_REGISTRY_DIR", custom_registry)

        # Import and create registry
        from ai_trading.model_registry import ModelRegistry

        ModelRegistry()

        # Check that directory was created
        assert os.path.exists(custom_registry), f"Directory {custom_registry} should exist"


def test_pytest_default_model_registry_is_isolated_from_runtime():
    from ai_trading.model_registry import ModelRegistry
    from ai_trading.paths import MODELS_DIR

    registry = ModelRegistry()

    assert registry.base_path != MODELS_DIR.resolve()
    assert "ai-trading-pytest-registry-" in str(registry.base_path)


if __name__ == "__main__":
    test_walkforward_artifacts_directory()
    test_model_registry_directory()
