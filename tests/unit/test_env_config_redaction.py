import logging

from ai_trading import main
from ai_trading.logging.redact import _ENV_MASK, _SENSITIVE_ENV, redact_env


def test_startup_logs_drop_secrets(caplog, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "AK123456789")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "SK987654321")
    monkeypatch.setenv("ALPACA_API_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("ALPACA_DATA_FEED", "iex")
    monkeypatch.setenv("WEBHOOK_SECRET", "HOOK-SECRET")
    monkeypatch.setenv("CAPITAL_CAP", "0.5")
    monkeypatch.setenv("DOLLAR_RISK_LIMIT", "1000")

    caplog.set_level(logging.INFO)
    main._fail_fast_env()
    env_log = next(rec for rec in caplog.records if rec.getMessage() == "ENV_CONFIG_LOADED")
    joined = str(env_log.__dict__)
    for key in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "WEBHOOK_SECRET"):
        assert key not in env_log.__dict__
        assert key not in joined
    assert _ENV_MASK not in joined
    assert "AK123456789" not in joined
    assert "SK987654321" not in joined
    assert "HOOK-SECRET" not in joined


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
