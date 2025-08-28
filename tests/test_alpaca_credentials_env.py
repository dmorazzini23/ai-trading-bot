"""Test ALPACA_* credential handling."""

import os
from unittest.mock import patch

import pytest
from ai_trading.config.management import _resolve_alpaca_env, validate_alpaca_credentials


class TestAlpacaCredentials:
    """Ensure only ALPACA_* variables are supported."""

    def test_resolve_all_alpaca_vars(self) -> None:
        env_vars = {
            "ALPACA_API_KEY": "fake_alpaca_key_not_real",
            "ALPACA_SECRET_KEY": "fake_alpaca_secret_not_real",
            "ALPACA_BASE_URL": "https://custom.alpaca.markets",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            api_key, secret_key, base_url = _resolve_alpaca_env()
            assert api_key == "fake_alpaca_key_not_real"
            assert secret_key == "fake_alpaca_secret_not_real"
            assert base_url == "https://custom.alpaca.markets"

    def test_resolve_alias_api_url(self) -> None:
        env_vars = {
            "ALPACA_API_KEY": "alias_key",
            "ALPACA_SECRET_KEY": "alias_secret",
            "ALPACA_API_URL": "https://alias-api.alpaca.markets",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            api_key, secret_key, base_url = _resolve_alpaca_env()
            assert api_key == "alias_key"
            assert secret_key == "alias_secret"
            assert base_url == "https://alias-api.alpaca.markets"

    def test_default_base_url_when_missing(self) -> None:
        env_vars = {
            "ALPACA_API_KEY": "key",
            "ALPACA_SECRET_KEY": "secret",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            api_key, secret_key, base_url = _resolve_alpaca_env()
            assert api_key == "key"
            assert secret_key == "secret"
            assert base_url == "https://paper-api.alpaca.markets"

    def test_missing_credentials(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            api_key, secret_key, base_url = _resolve_alpaca_env()
            assert api_key is None
            assert secret_key is None
            assert base_url == "https://paper-api.alpaca.markets"

    def test_partial_credentials_api_key_only(self) -> None:
        env_vars = {"ALPACA_API_KEY": "partial_key"}
        with patch.dict(os.environ, env_vars, clear=True):
            api_key, secret_key, base_url = _resolve_alpaca_env()
            assert api_key == "partial_key"
            assert secret_key is None
            assert base_url == "https://paper-api.alpaca.markets"

    def test_partial_credentials_secret_key_only(self) -> None:
        env_vars = {"ALPACA_SECRET_KEY": "partial_secret"}
        with patch.dict(os.environ, env_vars, clear=True):
            api_key, secret_key, base_url = _resolve_alpaca_env()
            assert api_key is None
            assert secret_key == "partial_secret"
            assert base_url == "https://paper-api.alpaca.markets"

    @patch("ai_trading.config.management.TESTING", True)
    def test_validate_skip_in_testing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            validate_alpaca_credentials()  # Should not raise

    @patch("ai_trading.config.management.TESTING", False)
    def test_validate_missing_production(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError) as exc_info:
                validate_alpaca_credentials()
            assert "ALPACA_API_KEY" in str(exc_info.value)

    @patch("ai_trading.config.management.TESTING", False)
    def test_validate_success(self) -> None:
        env_vars = {
            "ALPACA_API_KEY": "valid_key_123",
            "ALPACA_SECRET_KEY": "valid_secret_456",
            "ALPACA_API_URL": "https://api.alpaca.markets",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            validate_alpaca_credentials()  # Should not raise

    @patch("ai_trading.config.management.TESTING", False)
    def test_validate_partial_key_only(self) -> None:
        env_vars = {"ALPACA_API_KEY": "partial_key"}
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(RuntimeError):
                validate_alpaca_credentials()

    @patch("ai_trading.config.management.TESTING", False)
    def test_validate_partial_secret_only(self) -> None:
        env_vars = {"ALPACA_SECRET_KEY": "partial_secret"}
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(RuntimeError):
                validate_alpaca_credentials()
