import importlib
import os
import sys
from pathlib import Path

import pytest

import config


def test_get_env_required_missing(monkeypatch):
    """get_env raises when required variable is absent."""
    with pytest.raises(RuntimeError):
        config.get_env("FOO_BAR_MISSING", required=True)


def test_require_env_vars_failure(monkeypatch, caplog):
    """_require_env_vars logs and raises for missing keys."""
    caplog.set_level("CRITICAL")
    with pytest.raises(RuntimeError):
        config._require_env_vars("NEEDED_VAR")
    assert "Missing required environment variables" in caplog.text


def test_validate_environment_failure(monkeypatch):
    """validate_environment raises when vars missing."""
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        config.validate_environment()


def test_validate_alpaca_credentials_missing(monkeypatch):
    """validate_alpaca_credentials raises when credentials absent."""
    monkeypatch.setattr(config, "ALPACA_API_KEY", "", raising=False)
    monkeypatch.setattr(config, "ALPACA_SECRET_KEY", "", raising=False)
    monkeypatch.setattr(config, "ALPACA_BASE_URL", "", raising=False)
    with pytest.raises(RuntimeError):
        config.validate_alpaca_credentials()
