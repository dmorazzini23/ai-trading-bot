"""
Test dual schema credential support.

Tests that both ALPACA_* and APCA_* environment variable naming schemes
are supported for Alpaca credentials, with proper precedence and validation.
"""

import os
from unittest.mock import patch

import pytest
from ai_trading.config.management import (
    _resolve_alpaca_env,
    _warn_duplicate_env_keys,
    validate_alpaca_credentials,
)


class TestDualSchemaCredentials:
    """Test dual credential schema support."""

    def test_alpaca_schema_only(self):
        """Test using only ALPACA_* environment variables."""
        env_vars = {
            "ALPACA_API_KEY": "fake_alpaca_key_not_real",
            "ALPACA_SECRET_KEY": "fake_alpaca_secret_not_real",
            "ALPACA_BASE_URL": "https://paper-api.alpaca.markets"
        }

        with patch.dict(os.environ, env_vars, clear=True):
            api_key, secret_key, base_url = _resolve_alpaca_env()

            assert api_key == "fake_alpaca_key_not_real"
            assert secret_key == "fake_alpaca_secret_not_real"
            assert base_url == "https://paper-api.alpaca.markets"

    def test_apca_schema_only(self):
        """Test using only APCA_* environment variables."""
        env_vars = {
            "APCA_API_KEY_ID": "fake_apca_key_not_real",
            "APCA_API_SECRET_KEY": "fake_apca_secret_not_real",
            "APCA_API_BASE_URL": "https://api.alpaca.markets"
        }

        with patch.dict(os.environ, env_vars, clear=True):
            api_key, secret_key, base_url = _resolve_alpaca_env()

            assert api_key == "fake_apca_key_not_real"
            assert secret_key == "fake_apca_secret_not_real"
            assert base_url == "https://api.alpaca.markets"

    def test_alpaca_precedence_over_apca(self):
        """Test that ALPACA_* variables take precedence over APCA_*."""
        env_vars = {
            "ALPACA_API_KEY": "fake_alpaca_key_priority_not_real",
            "ALPACA_SECRET_KEY": "fake_alpaca_secret_priority_not_real",
            "ALPACA_BASE_URL": "https://alpaca-priority.com",
            "APCA_API_KEY_ID": "fake_apca_key_fallback_not_real",
            "APCA_API_SECRET_KEY": "fake_apca_secret_fallback_not_real",
            "APCA_API_BASE_URL": "https://apca-fallback.com"
        }

        with patch.dict(os.environ, env_vars, clear=True):
            api_key, secret_key, base_url = _resolve_alpaca_env()

            # Should use ALPACA_* values
            assert api_key == "fake_alpaca_key_priority_not_real"
            assert secret_key == "fake_alpaca_secret_priority_not_real"
            assert base_url == "https://alpaca-priority.com"

    def test_mixed_schema_alpaca_key_apca_secret(self):
        """Test mixed schema with ALPACA key and APCA secret."""
        env_vars = {
            "ALPACA_API_KEY": "fake_alpaca_key_mixed_not_real",
            "APCA_API_SECRET_KEY": "fake_apca_secret_mixed_not_real"
        }

        with patch.dict(os.environ, env_vars, clear=True):
            api_key, secret_key, base_url = _resolve_alpaca_env()

            assert api_key == "fake_alpaca_key_mixed_not_real"
            assert secret_key == "fake_apca_secret_mixed_not_real"
            assert base_url == "https://paper-api.alpaca.markets"  # Default

    def test_mixed_schema_apca_key_alpaca_secret(self):
        """Test mixed schema with APCA key and ALPACA secret."""
        env_vars = {
            "APCA_API_KEY_ID": "fake_apca_key_mixed_not_real",
            "ALPACA_SECRET_KEY": "fake_alpaca_secret_mixed_not_real"
        }

        with patch.dict(os.environ, env_vars, clear=True):
            api_key, secret_key, base_url = _resolve_alpaca_env()

            assert api_key == "fake_apca_key_mixed_not_real"
            assert secret_key == "fake_alpaca_secret_mixed_not_real"
            assert base_url == "https://paper-api.alpaca.markets"  # Default

    def test_default_base_url_when_missing(self):
        """Test that default base URL is set when none provided."""
        env_vars = {
            "ALPACA_API_KEY": "test_key",
            "ALPACA_SECRET_KEY": "test_secret"
            # No base URL provided
        }

        with patch.dict(os.environ, env_vars, clear=True):
            api_key, secret_key, base_url = _resolve_alpaca_env()

            assert api_key == "test_key"
            assert secret_key == "test_secret"
            assert base_url == "https://paper-api.alpaca.markets"

    def test_missing_credentials(self):
        """Test behavior when credentials are missing."""
        with patch.dict(os.environ, {}, clear=True):
            api_key, secret_key, base_url = _resolve_alpaca_env()

            assert api_key is None
            assert secret_key is None
            assert base_url == "https://paper-api.alpaca.markets"  # Default still set

    def test_partial_credentials_api_key_only(self):
        """Test behavior with only API key present."""
        env_vars = {
            "ALPACA_API_KEY": "partial_key"
        }

        with patch.dict(os.environ, env_vars, clear=True):
            api_key, secret_key, base_url = _resolve_alpaca_env()

            assert api_key == "partial_key"
            assert secret_key is None
            assert base_url == "https://paper-api.alpaca.markets"

    def test_partial_credentials_secret_key_only(self):
        """Test behavior with only secret key present."""
        env_vars = {
            "ALPACA_SECRET_KEY": "partial_secret"
        }

        with patch.dict(os.environ, env_vars, clear=True):
            api_key, secret_key, base_url = _resolve_alpaca_env()

            assert api_key is None
            assert secret_key == "partial_secret"
            assert base_url == "https://paper-api.alpaca.markets"

    @patch('ai_trading.config.management.logger')
    def test_warn_duplicate_env_keys_conflicting(self, mock_logger):
        """Test warning when duplicate keys have conflicting values."""
        env_vars = {
            "ALPACA_API_KEY": "alpaca_value",
            "APCA_API_KEY_ID": "apca_value",  # Different value
            "ALPACA_SECRET_KEY": "same_secret",
            "APCA_API_SECRET_KEY": "same_secret"  # Same value, no warning
        }

        with patch.dict(os.environ, env_vars, clear=True):
            _warn_duplicate_env_keys()

            # Should warn about API key conflict but not secret key
            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "ALPACA_API_KEY" in warning_call
            assert "APCA_API_KEY_ID" in warning_call
            assert "different values" in warning_call

    @patch('ai_trading.config.management.logger')
    def test_warn_duplicate_env_keys_no_conflict(self, mock_logger):
        """Test no warning when duplicate keys have same values."""
        env_vars = {
            "ALPACA_API_KEY": "same_value",
            "APCA_API_KEY_ID": "same_value",
            "ALPACA_SECRET_KEY": "same_secret",
            "APCA_API_SECRET_KEY": "same_secret"
        }

        with patch.dict(os.environ, env_vars, clear=True):
            _warn_duplicate_env_keys()

            # Should not warn since values are identical
            mock_logger.warning.assert_not_called()

    @patch('ai_trading.config.management.TESTING', True)
    def test_validate_alpaca_credentials_skip_in_testing(self):
        """Test that validation is skipped in testing mode."""
        with patch.dict(os.environ, {}, clear=True):
            # Should not raise exception in testing mode
            validate_alpaca_credentials()

    @patch('ai_trading.config.management.TESTING', False)
    @patch('ai_trading.config.management.logger')
    def test_validate_alpaca_credentials_missing_production(self, mock_logger):
        """Test that validation fails in production mode with missing credentials."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError) as exc_info:
                validate_alpaca_credentials()

            assert "Missing Alpaca credentials" in str(exc_info.value)
            assert "ALPACA_API_KEY" in str(exc_info.value)
            assert "APCA_API_KEY_ID" in str(exc_info.value)

    @patch('ai_trading.config.management.TESTING', False)
    @patch('ai_trading.config.management.logger')
    def test_validate_alpaca_credentials_success_alpaca_schema(self, mock_logger):
        """Test successful validation with ALPACA_* schema."""
        env_vars = {
            "ALPACA_API_KEY": "valid_key_123",
            "ALPACA_SECRET_KEY": "valid_secret_456"
        }

        with patch.dict(os.environ, env_vars, clear=True):
            # Should not raise exception
            validate_alpaca_credentials()

            # Should log success
            mock_logger.info.assert_called_with("Alpaca credentials resolved successfully")

    @patch('ai_trading.config.management.TESTING', False)
    @patch('ai_trading.config.management.logger')
    def test_validate_alpaca_credentials_success_apca_schema(self, mock_logger):
        """Test successful validation with APCA_* schema."""
        env_vars = {
            "APCA_API_KEY_ID": "valid_key_789",
            "APCA_API_SECRET_KEY": "valid_secret_012"
        }

        with patch.dict(os.environ, env_vars, clear=True):
            # Should not raise exception
            validate_alpaca_credentials()

            # Should log success
            mock_logger.info.assert_called_with("Alpaca credentials resolved successfully")

    @patch('ai_trading.config.management.TESTING', False)
    @patch('ai_trading.config.management.logger')
    def test_validate_alpaca_credentials_partial_key_only(self, mock_logger):
        """Test validation failure with only API key."""
        env_vars = {
            "ALPACA_API_KEY": "partial_key"
            # Missing secret key
        }

        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(RuntimeError) as exc_info:
                validate_alpaca_credentials()

            assert "Missing Alpaca credentials" in str(exc_info.value)

    @patch('ai_trading.config.management.TESTING', False)
    @patch('ai_trading.config.management.logger')
    def test_validate_alpaca_credentials_partial_secret_only(self, mock_logger):
        """Test validation failure with only secret key."""
        env_vars = {
            "ALPACA_SECRET_KEY": "partial_secret"
            # Missing API key
        }

        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(RuntimeError) as exc_info:
                validate_alpaca_credentials()

            assert "Missing Alpaca credentials" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])
