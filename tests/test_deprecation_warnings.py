"""
Test deprecation warnings for root module imports.
"""
import warnings


def test_bot_engine_deprecation_warning():
    """Importing bot_engine emits DeprecationWarning."""  # AI-AGENT-REF
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import ai_trading.core.bot_engine  # noqa: F401
        assert any(issubclass(w.category, DeprecationWarning) for w in w)


def test_data_fetcher_deprecation_warning():
    """Test that importing data_fetcher shows deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import ai_trading.data_fetcher  # noqa: F401

        # Check that a deprecation warning was raised
        assert len(w) >= 1
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
        assert any("data_fetcher.py is deprecated" in str(warning.message) for warning in w)


def test_runner_deprecation_warning():
    """Test that importing runner shows deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import ai_trading.runner  # noqa: F401

        # Check that a deprecation warning was raised
        assert len(w) >= 1
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
        assert any("runner.py is deprecated" in str(warning.message) for warning in w)


def test_alpaca_api_deprecation_warning():
    """Test canonical alpaca_api import emits no warnings."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from ai_trading import alpaca_api  # noqa: F401  # AI-AGENT-REF: canonical import

        # Ensure no deprecation warning is raised for packaged import
        assert not w

# AI-AGENT-REF: Test deprecation warnings for root module shims
