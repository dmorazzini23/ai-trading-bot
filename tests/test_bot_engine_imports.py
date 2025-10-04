"""Tests for bot engine import behavior using canonical modules."""

import builtins
import importlib
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest


class TestBotEngineImports:
    """Test import logic in bot_engine using canonical modules."""

    _original_modules = dict(sys.modules)

    @staticmethod
    @contextmanager
    def _safe_patch_modules(values: dict[str, object], *, clear: bool = False):
        with patch.dict(sys.modules, values, clear=clear) as ctx:
            if clear:
                for name, module in TestBotEngineImports._original_modules.items():
                    if module is not None and name not in sys.modules:
                        sys.modules[name] = module
            yield ctx

    def test_model_pipeline_import(self):
        """``model_pipeline`` should import from package path."""
        mock_pipeline = MagicMock()
        mock_pipeline.model_pipeline = "mock_model_pipeline"
        with self._safe_patch_modules({"ai_trading.pipeline": mock_pipeline}):
            with patch("builtins.__import__") as mock_import:
                def side_effect(name, *args, **kwargs):
                    if name == "ai_trading.pipeline":
                        return mock_pipeline
                    return MagicMock()

                mock_import.side_effect = side_effect
                from ai_trading.pipeline import model_pipeline  # type: ignore
                assert model_pipeline == "mock_model_pipeline"

    def test_import_error_when_module_missing(self):
        """Importing pipeline should raise if module absent."""
        with self._safe_patch_modules({}, clear=True):
            with pytest.raises(ImportError):
                from ai_trading.pipeline import model_pipeline  # type: ignore

    def test_sys_module_restored_after_clear(self):
        """Clearing ``sys.modules`` should keep essential modules accessible."""

        with self._safe_patch_modules({}, clear=True):
            optimizer = importlib.import_module("ai_trading.portfolio.optimizer")

            assert optimizer  # imported without NameError due to missing sys
            assert sys.modules.get("sys") is sys
            assert sys.modules.get("builtins") is builtins
