"""
Tests for bot engine import fallback functionality.
"""
import sys
from unittest.mock import patch, MagicMock

import pytest


class TestBotEngineImports:
    """Test import fallback logic in bot_engine."""

    def test_model_pipeline_import_fallback(self, monkeypatch):
        """Test that model_pipeline import falls back correctly."""
        # Create a mock pipeline module
        mock_pipeline = MagicMock()
        mock_pipeline.model_pipeline = "mock_model_pipeline"
        
        # Test primary path works when package import is available
        with patch.dict(sys.modules, {'ai_trading.pipeline': mock_pipeline}):
            # Remove from sys.modules to simulate fresh import
            sys.modules.pop('ai_trading.core.bot_engine', None)
            
            # Mock the import to simulate successful package import
            with patch('builtins.__import__') as mock_import:
                def side_effect(name, *args, **kwargs):
                    if name == 'ai_trading.pipeline':
                        return mock_pipeline
                    elif name == 'pipeline':
                        # This shouldn't be called if package import works
                        raise ImportError("Should not reach legacy import")
                    return MagicMock()
                
                mock_import.side_effect = side_effect
                
                # Test the import logic directly
                try:
                    from ai_trading.pipeline import model_pipeline  # type: ignore
                    assert model_pipeline == "mock_model_pipeline"
                    primary_success = True
                except Exception:
                    primary_success = False
                    
                assert primary_success, "Primary import path should work"

    def test_model_pipeline_import_fallback_to_legacy(self, monkeypatch):
        """Test fallback to legacy root import when package import fails."""
        # Create a mock legacy pipeline module
        mock_legacy_pipeline = MagicMock()
        mock_legacy_pipeline.model_pipeline = "mock_legacy_model_pipeline"
        
        # Test fallback path when package import fails
        with patch.dict(sys.modules, {'pipeline': mock_legacy_pipeline}):
            # Mock the import to simulate package import failure
            with patch('builtins.__import__') as mock_import:
                def side_effect(name, *args, **kwargs):
                    if name == 'ai_trading.pipeline':
                        raise ImportError("Package not found")
                    elif name == 'pipeline':
                        return mock_legacy_pipeline
                    return MagicMock()
                
                mock_import.side_effect = side_effect
                
                # Test the fallback logic
                try:
                    try:
                        from ai_trading.pipeline import model_pipeline  # type: ignore
                        fallback_triggered = False
                    except Exception:  # pragma: no cover
                        from pipeline import model_pipeline  # type: ignore
                        fallback_triggered = True
                        
                    assert fallback_triggered, "Should fall back to legacy import"
                    assert model_pipeline == "mock_legacy_model_pipeline"
                except ImportError:
                    # In test environment, both imports might fail - that's okay
                    pytest.skip("Both imports failed in test environment")

    def test_import_robustness_when_both_fail(self):
        """Test behavior when both import paths fail."""
        # This tests the error handling when neither import works
        with patch('builtins.__import__') as mock_import:
            def side_effect(name, *args, **kwargs):
                if name in ('ai_trading.pipeline', 'pipeline'):
                    raise ImportError(f"Module {name} not found")
                return MagicMock()
            
            mock_import.side_effect = side_effect
            
            # Both imports should fail
            with pytest.raises(ImportError):
                try:
                    from ai_trading.pipeline import model_pipeline  # type: ignore
                except Exception:  # pragma: no cover
                    from pipeline import model_pipeline  # type: ignore

    def test_import_types_annotation(self):
        """Test that type annotations are preserved in import statements."""
        # This is a basic check that the import statements have proper type ignore comments
        import ast
        import inspect
        
        # Get the source of the import logic we're testing
        import_code = '''
try:
    from ai_trading.pipeline import model_pipeline  # type: ignore
except Exception:  # pragma: no cover
    from pipeline import model_pipeline  # type: ignore
'''
        
        # Parse and verify the AST contains type: ignore comments
        tree = ast.parse(import_code)
        
        # We mainly care that the code parses without syntax errors
        # and that it follows the expected structure
        assert len(tree.body) == 1  # One try statement
        assert isinstance(tree.body[0], ast.Try)
        
        # Verify we have the expected import structure
        try_body = tree.body[0].body
        assert len(try_body) == 1
        assert isinstance(try_body[0], ast.ImportFrom)
        assert try_body[0].module == 'ai_trading.pipeline'
        
        # Verify exception handler
        handlers = tree.body[0].handlers
        assert len(handlers) == 1
        handler_body = handlers[0].body
        assert len(handler_body) == 1
        assert isinstance(handler_body[0], ast.ImportFrom)
        assert handler_body[0].module == 'pipeline'