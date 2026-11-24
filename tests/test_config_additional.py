
import importlib
import os

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




def test_validate_alpaca_credentials_missing(monkeypatch):
    """validate_alpaca_credentials raises when credentials absent."""
    monkeypatch.setattr(config.management, "TESTING", False, raising=False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("ALPACA_TRADING_BASE_URL", raising=False)
    monkeypatch.delenv("ALPACA_API_URL", raising=False)
    monkeypatch.delenv("ALPACA_BASE_URL", raising=False)
    with pytest.raises(RuntimeError):
        config.validate_alpaca_credentials()


def test_validate_alpaca_credentials_reads_runtime_env(monkeypatch):
    """Credential validation should use current environment values."""
    monkeypatch.setattr(config.management, "TESTING", False, raising=False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.setenv("ALPACA_API_KEY", "runtime-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "runtime-secret")
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets")

    config.validate_alpaca_credentials()


def test_validate_alpaca_credentials_reads_testing_at_call_time(monkeypatch):
    """TESTING changes after import should affect credential validation."""
    monkeypatch.setattr(config.management, "TESTING", False, raising=False)
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("ALPACA_TRADING_BASE_URL", raising=False)
    monkeypatch.delenv("ALPACA_API_URL", raising=False)
    monkeypatch.delenv("ALPACA_BASE_URL", raising=False)

    monkeypatch.setenv("TESTING", "1")
    config.validate_alpaca_credentials()

    monkeypatch.setenv("TESTING", "0")
    with pytest.raises(RuntimeError):
        config.validate_alpaca_credentials()


def test_reload_env_clears_cached_settings(monkeypatch):
    """reload_env should invalidate cached config settings objects."""
    # Other config tests reimport the package; use the current registered
    # module so reload_env invalidates the same cache this test reads.
    current_config = importlib.import_module("ai_trading.config")
    monkeypatch.setenv("AI_TRADING_CAPITAL_CAP", "0.31")
    current_config._reset_cached_settings()
    first = current_config.get_settings()

    monkeypatch.setenv("AI_TRADING_CAPITAL_CAP", "0.44")
    # Reloading the package elsewhere can leave its parent attribute pointing
    # at an older module even though sys.modules holds the current one.
    monkeypatch.setattr(importlib.import_module("ai_trading"), "config", object())
    current_config.reload_env(path=None)
    second = current_config.get_settings()

    assert second is not first
    assert float(getattr(second, "capital_cap", 0.0)) == pytest.approx(0.44)


def test_reload_env_refreshes_exported_runtime_constants(monkeypatch, tmp_path):
    # reload_env writes the dotenv values into os.environ. Snapshot and restore
    # them: a leaked EXECUTION_MODE=sim changes the launch profile, and with it
    # short permissions, for every later test. (monkeypatch.delenv records
    # nothing for an absent key, so it cannot undo this.)
    keys = (
        "EXECUTION_MODE",
        "DATA_FEED_INTRADAY",
        "ALPACA_EXECUTION_FEED",
        "ALPACA_ALLOW_SIP",
        "ALPACA_HAS_SIP",
        "AI_TRADING_SAFE_MODE_ALLOW_PAPER",
    )
    saved = {key: os.environ.get(key) for key in keys}
    try:
        env_path = tmp_path / ".env"
        env_path.write_text(
            "\n".join(
                [
                    "EXECUTION_MODE=paper",
                    "DATA_FEED_INTRADAY=sip",
                    "ALPACA_EXECUTION_FEED=sip",
                    "ALPACA_ALLOW_SIP=1",
                    "ALPACA_HAS_SIP=1",
                    "AI_TRADING_SAFE_MODE_ALLOW_PAPER=1",
                ]
            ),
            encoding="utf-8",
        )

        config.reload_env(path=env_path)

        assert config.EXECUTION_MODE == "paper"
        assert config.DATA_FEED_INTRADAY == "sip"
        assert config.ALPACA_EXECUTION_FEED == "sip"
        assert config.SAFE_MODE_ALLOW_PAPER is True

        env_path.write_text(
            "\n".join(
                [
                    "EXECUTION_MODE=sim",
                    "DATA_FEED_INTRADAY=iex",
                    "ALPACA_EXECUTION_FEED=iex",
                    "AI_TRADING_SAFE_MODE_ALLOW_PAPER=0",
                ]
            ),
            encoding="utf-8",
        )

        config.reload_env(path=env_path)

        assert config.EXECUTION_MODE == "sim"
        assert config.DATA_FEED_INTRADAY == "iex"
        assert config.ALPACA_EXECUTION_FEED == "iex"
        assert config.SAFE_MODE_ALLOW_PAPER is False
    finally:
        # Then re-derive the exported constants and cached settings from the
        # restored environment so later tests do not see sim/iex.
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        config.reload_env()


