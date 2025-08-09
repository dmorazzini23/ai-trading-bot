"""
Test deprecation warnings for root module imports.
"""
import warnings

def test_bot_engine_deprecation_warning():
    """Test that importing bot_engine shows deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import bot_engine  # noqa: F401
        
        # Check that a deprecation warning was raised
        assert len(w) >= 1
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
        assert any("bot_engine.py is deprecated" in str(warning.message) for warning in w)

def test_data_fetcher_deprecation_warning():
    """Test that importing data_fetcher shows deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import data_fetcher  # noqa: F401
        
        # Check that a deprecation warning was raised
        assert len(w) >= 1
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
        assert any("data_fetcher.py is deprecated" in str(warning.message) for warning in w)

def test_runner_deprecation_warning():
    """Test that importing runner shows deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import runner  # noqa: F401
        
        # Check that a deprecation warning was raised
        assert len(w) >= 1
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
        assert any("runner.py is deprecated" in str(warning.message) for warning in w)

def test_alpaca_api_deprecation_warning():
    """Test that importing alpaca_api shows deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import alpaca_api  # noqa: F401
        
        # Check that a deprecation warning was raised
        assert len(w) >= 1
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
        assert any("alpaca_api.py is deprecated" in str(warning.message) for warning in w)

# AI-AGENT-REF: Test deprecation warnings for root module shims