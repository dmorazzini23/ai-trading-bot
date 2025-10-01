import importlib
import logging

config_pkg = importlib.import_module("ai_trading.config")
if not hasattr(config_pkg, "get_settings"):
    settings_mod = importlib.import_module("ai_trading.config.settings")
    config_pkg.get_settings = settings_mod.get_settings

from ai_trading import main
from ai_trading.config.management import _resolve_alpaca_env, validate_required_env
from ai_trading.logging.redact import _ENV_MASK, _SENSITIVE_ENV, redact_env
from ai_trading.env.config_redaction import redact_config_env


def test_startup_logs_drop_secrets(caplog, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "AK123456789")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "SK987654321")
    monkeypatch.setenv("ALPACA_API_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("ALPACA_DATA_FEED", "iex")
    monkeypatch.setenv("WEBHOOK_SECRET", "HOOK-SECRET")
    monkeypatch.setenv("CAPITAL_CAP", "0.5")
    monkeypatch.setenv("DOLLAR_RISK_LIMIT", "0.5")

    caplog.set_level(logging.INFO)
    main._fail_fast_env()
    env_log = next(rec for rec in caplog.records if rec.getMessage() == "ENV_CONFIG_LOADED")
    joined = str(env_log.__dict__)
    for key in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "WEBHOOK_SECRET"):
        assert key not in env_log.__dict__
        assert key not in joined
    assert "AK123456789" not in joined
    assert "SK987654321" not in joined
    assert "HOOK-SECRET" not in joined
    assert env_log.ALPACA_DATA_FEED == "iex"


def test_redact_env_masks_all_keys_by_default():
    payload = {k: f"{k}_VALUE" for k in _SENSITIVE_ENV}
    redacted = redact_env(payload)
    assert all(redacted[k] == _ENV_MASK for k in payload)
    assert set(redacted) == set(payload)


def test_redact_env_drop_removes_keys():
    payload = {k: f"{k}_VALUE" for k in _SENSITIVE_ENV}
    redacted = redact_env(payload, drop=True)
    for k in _SENSITIVE_ENV:
        assert k not in redacted


def test_redact_config_env_alias_mapped():
    payload = {"ALPACA_BASE_URL": "https://alias-api.alpaca.markets"}
    redacted = redact_config_env(payload)
    assert redacted["ALPACA_API_URL"] == "https://alias-api.alpaca.markets"
    assert "ALPACA_BASE_URL" not in redacted


def test_redact_config_env_does_not_alias_apca():
    payload = {"APCA_API_BASE_URL": "https://legacy-api.alpaca.markets"}
    redacted = redact_config_env(payload)
    assert "ALPACA_API_URL" not in redacted
    assert redacted["APCA_API_BASE_URL"] == "https://legacy-api.alpaca.markets"


def test_validate_required_env_handles_feed(monkeypatch):
    env = {
        "ALPACA_API_KEY": "key",
        "ALPACA_SECRET_KEY": "secret",
        "ALPACA_DATA_FEED": "iex",
        "ALPACA_API_URL": "https://paper-api.alpaca.markets",
        "WEBHOOK_SECRET": "hook",
        "CAPITAL_CAP": "1.0",
        "DOLLAR_RISK_LIMIT": "0.5",
    }
    redacted = validate_required_env(
        ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_DATA_FEED"), env=env
    )
    assert redacted == {
        "ALPACA_API_KEY": "***",
        "ALPACA_SECRET_KEY": "***",
        "ALPACA_DATA_FEED": "***",
    }


def test_base_url_alias_logged(caplog, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "AK123456789")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "SK987654321")
    monkeypatch.delenv("ALPACA_API_URL", raising=False)
    monkeypatch.setenv("ALPACA_BASE_URL", "https://alias-api.alpaca.markets")
    monkeypatch.setenv("ALPACA_DATA_FEED", "iex")
    monkeypatch.setenv("WEBHOOK_SECRET", "HOOK-SECRET")
    monkeypatch.setenv("CAPITAL_CAP", "0.5")
    monkeypatch.setenv("DOLLAR_RISK_LIMIT", "0.5")

    caplog.set_level(logging.INFO)
    main._fail_fast_env()
    env_log = next(rec for rec in caplog.records if rec.getMessage() == "ENV_CONFIG_LOADED")
    assert env_log.ALPACA_API_URL == "https://alias-api.alpaca.markets"
    assert not hasattr(env_log, "ALPACA_BASE_URL")

    redacted = redact_env({"ALPACA_BASE_URL": "https://alias-api.alpaca.markets"})
    assert redacted["ALPACA_API_URL"] == "https://alias-api.alpaca.markets"
    assert "ALPACA_BASE_URL" not in redacted

    _, _, resolved_base_url = _resolve_alpaca_env()
    assert resolved_base_url == "https://alias-api.alpaca.markets"
