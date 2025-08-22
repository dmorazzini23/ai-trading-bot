"""Test DISABLE_DAILY_RETRAIN parsing for various values."""

import os


def test_disable_daily_retrain_env_parsing():
    """Test DISABLE_DAILY_RETRAIN parsing for various values."""
    test_cases = [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("1", True),
        ("false", False),
        ("False", False),
        ("FALSE", False),
        ("0", False),
        ("", False),  # empty string should default to False
        ("invalid", False),  # invalid values should default to False
    ]

    for env_value, expected in test_cases:
        # Set the environment variable
        os.environ["DISABLE_DAILY_RETRAIN"] = env_value
        os.environ["TESTING"] = "1"  # Enable testing mode

        # Clear module cache to force re-import
        if 'config' in os.sys.modules:
            del os.sys.modules['config']

        # Import config module
        from ai_trading import config

        # Test the result
        actual = config.DISABLE_DAILY_RETRAIN
        assert actual == expected, f"For env value '{env_value}', expected {expected}, got {actual}"

        # Clean up
        if 'config' in os.sys.modules:
            del os.sys.modules['config']


def test_disable_daily_retrain_unset():
    """Test DISABLE_DAILY_RETRAIN when environment variable is unset."""
    # Remove the environment variable if it exists
    if "DISABLE_DAILY_RETRAIN" in os.environ:
        del os.environ["DISABLE_DAILY_RETRAIN"]

    os.environ["TESTING"] = "1"  # Enable testing mode

    # Clear module cache
    if 'config' in os.sys.modules:
        del os.sys.modules['config']

    # Import config module
    from ai_trading import config

    # Should default to False
    assert config.DISABLE_DAILY_RETRAIN == False

    # Clean up
    if 'config' in os.sys.modules:
        del os.sys.modules['config']


def test_disable_daily_retrain_fallback_settings():
    """Test DISABLE_DAILY_RETRAIN through fallback settings."""
    # Test the fallback _FallbackSettings class directly
    if 'config' in os.sys.modules:
        del os.sys.modules['config']

    os.environ["TESTING"] = "1"
    os.environ["DISABLE_DAILY_RETRAIN"] = "true"

    from ai_trading import config

    # Check that fallback settings work
    fallback = config._FallbackSettings()
    assert fallback.DISABLE_DAILY_RETRAIN == True

    os.environ["DISABLE_DAILY_RETRAIN"] = "false"
    fallback2 = config._FallbackSettings()
    assert fallback2.DISABLE_DAILY_RETRAIN == False


def teardown_module():
    """Clean up after tests."""
    # Remove test environment variables
    for var in ["DISABLE_DAILY_RETRAIN", "TESTING"]:
        if var in os.environ:
            del os.environ[var]

    # Clear module cache
    if 'config' in os.sys.modules:
        del os.sys.modules['config']
