#!/usr/bin/env python3
"""
Test artifacts directory creation and environment variable overrides.
"""

import os
import tempfile




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


