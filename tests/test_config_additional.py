
import pytest
from ai_trading import config


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
    monkeypatch.setattr(config.management, "TESTING", False, raising=False)
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_URL", raising=False)
    monkeypatch.delenv("ALPACA_BASE_URL", raising=False)
    with pytest.raises(RuntimeError):
        config.validate_alpaca_credentials()


def test_validate_alpaca_credentials_reads_runtime_env(monkeypatch):
    """Credential validation should use current environment values."""
    monkeypatch.setattr(config.management, "TESTING", False, raising=False)
    monkeypatch.setenv("ALPACA_API_KEY", "runtime-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "runtime-secret")
    monkeypatch.setenv("ALPACA_API_URL", "https://paper-api.alpaca.markets")

    config.validate_alpaca_credentials()


def test_reload_env_clears_cached_settings(monkeypatch):
    """reload_env should invalidate cached config settings objects."""
    monkeypatch.setenv("CAPITAL_CAP", "0.31")
    config._reset_cached_settings()
    first = config.get_settings()

    monkeypatch.setenv("CAPITAL_CAP", "0.44")
    config.reload_env(path=None)
    second = config.get_settings()

    assert second is not first
    assert float(getattr(second, "capital_cap", 0.0)) == pytest.approx(0.44)


def test_log_config_does_not_log(monkeypatch, caplog):
    monkeypatch.setenv("ALPACA_API_KEY", "secret1234")
    caplog.set_level("INFO")
    config._CONFIG_LOGGED = False
    config.log_config(["ALPACA_API_KEY"])
    assert caplog.text == ""
    assert "secret1234" not in caplog.text
    assert config._CONFIG_LOGGED
