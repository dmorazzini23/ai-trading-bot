from __future__ import annotations

# Tests for redaction utility.  # AI-AGENT-REF
from ai_trading.logging.redact import redact


def test_redact_masks_nested() -> None:
    payload = {
        "apiKey": "abc",
        "nested": {"secret": "def", "token": "ghi"},
        "pwd": "not-masked",
        "Authorization": "Bearer abc",
    }
    out = redact(payload)
    assert out["apiKey"] == "***REDACTED***"
    assert out["nested"]["secret"] == "***REDACTED***"
    assert out["nested"]["token"] == "***REDACTED***"
    assert out["pwd"] == "***REDACTED***"
    assert out["Authorization"] == "***REDACTED***"


def test_redact_masks_connection_strings() -> None:
    payload = {
        "connection_string": "postgresql://user:secret@example.com/trading",
        "dsn": "postgresql://user:secret@example.com/trading",
    }

    out = redact(payload)

    assert out["connection_string"] == "***REDACTED***"
    assert out["dsn"] == "***REDACTED***"


def test_redact_env_masks_common_secret_aliases() -> None:
    from ai_trading.logging.redact import _ENV_MASK, redact_env

    env = {
        "APCA_API_KEY_ID": "key-id",
        "APCA_API_SECRET_KEY": "secret-key",
        "ALPACA_API_SECRET_KEY": "alias-secret",
        "AI_TRADING_OPENCLAW_HOOK_TOKEN": "hook-token",
        "AI_TRADING_SLACK_WEBHOOK_URL": "https://hooks.slack.example/services/T/B/C",
        "AI_TRADING_OPENCLAW_RUNTIME_WEBHOOK_URL": "https://hooks.example/runtime",
        "PLAIN_FLAG": "1",
    }

    out = redact_env(env)

    assert out["APCA_API_KEY_ID"] == _ENV_MASK
    assert out["APCA_API_SECRET_KEY"] == _ENV_MASK
    assert out["ALPACA_API_SECRET_KEY"] == _ENV_MASK
    assert out["AI_TRADING_OPENCLAW_HOOK_TOKEN"] == _ENV_MASK
    assert out["AI_TRADING_SLACK_WEBHOOK_URL"] == _ENV_MASK
    assert out["AI_TRADING_OPENCLAW_RUNTIME_WEBHOOK_URL"] == _ENV_MASK
    assert out["PLAIN_FLAG"] == "1"
