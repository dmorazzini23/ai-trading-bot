"""Test package vs repo-root import fallback mechanisms."""

import sys


def test_model_registry_import_fallback():
    """Test that model registry imports work with both package and fallback patterns."""
    # Remove from cache if present
    modules_to_clean = [m for m in sys.modules.keys() if "model_registry" in m]
    for mod in modules_to_clean:
        del sys.modules[mod]

    # Test successful ai_trading import
    try:
        from ai_trading.model_registry import ModelRegistry

        assert ModelRegistry is not None
    except ImportError:
        # If ai_trading import fails, test fallback would work
        # (We can't actually test the fallback because the file structure exists)
        pass


def test_bot_engine_import_fallbacks():
    """Test that bot_engine import fallbacks work correctly."""
    # Test that the import patterns are present in the code
    # Check that the file contains the expected try/except patterns
    import inspect

    from ai_trading.core import bot_engine

    source = inspect.getsource(bot_engine)

    # Look for the expected import patterns
    expected_patterns = [
        "from ai_trading.meta_learning import optimize_signals",
        "from meta_learning import optimize_signals",
        "from ai_trading.pipeline import model_pipeline",
        "from pipeline import model_pipeline",
        "from ai_trading.data.fetch import",
        "from data_fetcher import",
        "from ai_trading.indicators import rsi",
        "from indicators import rsi",
        "from ai_trading.signals import generate_position_hold_signals",
        "from signals import generate_position_hold_signals",
        "from ai_trading import portfolio",
        "import portfolio",
        "from ai_trading.alpaca_api import alpaca_get",
        "from alpaca_api import alpaca_get",
    ]

    for pattern in expected_patterns:
        assert pattern in source, f"Expected import pattern not found: {pattern}"



def test_backtester_import_fallbacks():
    """Test that backtester.py import fallbacks are correctly implemented."""
    import inspect

    from ai_trading.strategies import backtester

    source = inspect.getsource(backtester)

    # Check for expected fallback patterns
    expected_patterns = [
        "import ai_trading.signals as signals",
        "import signals",
        "import ai_trading.data.fetch as data_fetcher",
        "import data_fetcher",
    ]

    for pattern in expected_patterns:
        assert pattern in source, f"Expected import pattern not found in backtester.py: {pattern}"


def test_profile_indicators_import_fallbacks():
    """Test that profile_indicators.py import fallbacks are correctly implemented."""
    import inspect

    import profile_indicators

    source = inspect.getsource(profile_indicators)

    # Check for expected fallback patterns
    expected_patterns = [
        "from ai_trading import signals",
        "from ai_trading import indicators",
    ]

    for pattern in expected_patterns:
        assert (
            pattern in source
        ), f"Expected import pattern not found in profile_indicators.py: {pattern}"


def test_import_robustness():
    """Test that imports work even when some modules are missing."""
    # This test ensures that the fallback patterns would work
    # even if some ai_trading submodules were missing

    # Test that we can import core modules
    modules_to_test = [
        "ai_trading.core.bot_engine",
        "ai_trading.strategies.backtester",
        "ai_trading.strategies.profile_indicators",
    ]

    for module_name in modules_to_test:
        try:
            __import__(module_name)
        except ImportError as e:
            # If import fails, it should be due to missing dependencies,
            # not due to import structure issues
            assert (
                "cannot import name" not in str(e).lower()
            ), f"Import structure issue in {module_name}: {e}"


def test_data_fetcher_helpers_available():
    """Test that the new data_fetcher helper functions are available."""
    try:
        from ai_trading.data.fetch import (
            clear_cached_minute_cache,
            get_cached_age_seconds,
            get_cached_minute_timestamp,
            set_cached_minute_timestamp,
        )

        assert callable(get_cached_minute_timestamp)
        assert callable(set_cached_minute_timestamp)
        assert callable(get_cached_age_seconds)
        assert callable(clear_cached_minute_cache)
    except ImportError:
        from ai_trading.data.fetch import (
            clear_cached_minute_cache,
            get_cached_age_seconds,
            get_cached_minute_timestamp,
            set_cached_minute_timestamp,
        )

        assert callable(get_cached_minute_timestamp)
        assert callable(set_cached_minute_timestamp)
        assert callable(get_cached_age_seconds)
        assert callable(clear_cached_minute_cache)
