"""Test environment flag parsing for DISABLE_DAILY_RETRAIN."""

import os
import unittest
from unittest.mock import patch


class TestEnvFlags(unittest.TestCase):
    """Test environment variable parsing and boolean conversion."""

    def setUp(self):
        """Set up test by clearing relevant environment variables."""
        # Store original values
        self.original_env = {}
        env_vars = [
            "DISABLE_DAILY_RETRAIN",
            "ALPACA_API_KEY",
            "ALPACA_SECRET_KEY", 
            "ALPACA_BASE_URL",
            "WEBHOOK_SECRET",
            "TESTING"
        ]
        
        for var in env_vars:
            self.original_env[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]
        
        # Set testing mode to avoid validation errors
        os.environ["TESTING"] = "1"

    def tearDown(self):
        """Restore original environment variables."""
        for var, value in self.original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]

    def test_disable_daily_retrain_true_values(self):
        """Test DISABLE_DAILY_RETRAIN parsing for true values."""
        true_values = ["true", "True", "TRUE", "1"]
        
        for value in true_values:
            with self.subTest(value=value):
                os.environ["DISABLE_DAILY_RETRAIN"] = value
                
                # Force reload of config module
                import importlib
                import config
                importlib.reload(config)
                
                self.assertTrue(
                    config.DISABLE_DAILY_RETRAIN,
                    f"Expected True for DISABLE_DAILY_RETRAIN='{value}'"
                )

    def test_disable_daily_retrain_false_values(self):
        """Test DISABLE_DAILY_RETRAIN parsing for false values."""
        false_values = ["false", "False", "FALSE", "0", "no", "off"]
        
        for value in false_values:
            with self.subTest(value=value):
                os.environ["DISABLE_DAILY_RETRAIN"] = value
                
                # Force reload of config module
                import importlib
                import config
                importlib.reload(config)
                
                self.assertFalse(
                    config.DISABLE_DAILY_RETRAIN,
                    f"Expected False for DISABLE_DAILY_RETRAIN='{value}'"
                )

    def test_disable_daily_retrain_unset_default(self):
        """Test DISABLE_DAILY_RETRAIN default when unset."""
        # Ensure variable is not set
        if "DISABLE_DAILY_RETRAIN" in os.environ:
            del os.environ["DISABLE_DAILY_RETRAIN"]
        
        # Force reload of config module
        import importlib
        import config
        importlib.reload(config)
        
        self.assertFalse(
            config.DISABLE_DAILY_RETRAIN,
            "Expected False as default when DISABLE_DAILY_RETRAIN is unset"
        )

    def test_boolean_conversion_edge_cases(self):
        """Test edge cases in boolean conversion."""
        edge_cases = [
            ("", False),  # Empty string
            ("   true   ", True),  # Whitespace
            ("yes", False),  # Not in true list
            ("True1", False),  # Not exact match
        ]
        
        for value, expected in edge_cases:
            with self.subTest(value=repr(value), expected=expected):
                os.environ["DISABLE_DAILY_RETRAIN"] = value
                
                # Force reload of config module
                import importlib
                import config
                importlib.reload(config)
                
                self.assertEqual(
                    config.DISABLE_DAILY_RETRAIN,
                    expected,
                    f"Expected {expected} for DISABLE_DAILY_RETRAIN={repr(value)}"
                )

    def test_env_settings_fallback(self):
        """Test that fallback env_settings works correctly."""
        # Set a known value
        os.environ["DISABLE_DAILY_RETRAIN"] = "true"
        
        # Force reload to use fallback settings
        import importlib
        import config
        importlib.reload(config)
        
        # Test accessing through env_settings fallback
        self.assertTrue(
            config.env_settings.DISABLE_DAILY_RETRAIN,
            "env_settings.DISABLE_DAILY_RETRAIN should reflect environment variable"
        )

    def test_safe_defaults_multiple_flags(self):
        """Test safe defaults for multiple boolean flags."""
        boolean_flags = [
            ("SHADOW_MODE", True),  # Default True
            ("DISABLE_DAILY_RETRAIN", False),  # Default False
        ]
        
        # Clear all environment variables
        for flag, _ in boolean_flags:
            if flag in os.environ:
                del os.environ[flag]
        
        # Force reload
        import importlib
        import config
        importlib.reload(config)
        
        for flag, expected_default in boolean_flags:
            with self.subTest(flag=flag):
                actual_value = getattr(config, flag)
                self.assertEqual(
                    actual_value,
                    expected_default,
                    f"Expected {expected_default} as default for {flag}"
                )

    def test_config_key_consistency(self):
        """Test that config reads from correct environment key."""
        # Set the environment variable
        os.environ["DISABLE_DAILY_RETRAIN"] = "true"
        
        # Also set a wrong key to ensure it's not being used
        os.environ["DISABLE_RETRAIN"] = "false"  # Wrong key
        
        # Force reload
        import importlib
        import config
        importlib.reload(config)
        
        # Should read from correct key (DISABLE_DAILY_RETRAIN), not wrong key
        self.assertTrue(
            config.DISABLE_DAILY_RETRAIN,
            "Should read from DISABLE_DAILY_RETRAIN, not other similar keys"
        )

    def test_pydantic_fallback_behavior(self):
        """Test behavior when pydantic-settings is not available."""
        import importlib
        
        # Mock pydantic unavailability
        with patch.dict('sys.modules', {'pydantic_settings': None}):
            with patch('config._PYDANTIC_AVAILABLE', False):
                os.environ["DISABLE_DAILY_RETRAIN"] = "true"
                
                # Force reload without pydantic
                import config
                importlib.reload(config)
                
                # Should still work with fallback
                self.assertTrue(
                    config.DISABLE_DAILY_RETRAIN,
                    "Should work correctly with pydantic fallback"
                )


if __name__ == "__main__":
    unittest.main()