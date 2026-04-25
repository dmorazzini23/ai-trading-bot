from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from ai_trading import security


class _FakeAuditLogger:
    def __init__(self) -> None:
        self.records: list[str] = []
        self.handlers: list[Any] = [object()]

    def setLevel(self, _level: int) -> None:
        return None

    def info(self, message: str) -> None:
        self.records.append(message)


class _FakeLogger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def debug(self, message: str, *args: Any, **_kwargs: Any) -> None:
        self.messages.append(("debug", message % args if args else message))

    def info(self, message: str, *args: Any, **_kwargs: Any) -> None:
        self.messages.append(("info", message % args if args else message))

    def warning(self, message: str, *args: Any, **_kwargs: Any) -> None:
        self.messages.append(("warning", message % args if args else message))

    def error(self, message: str, *args: Any, **_kwargs: Any) -> None:
        self.messages.append(("error", message % args if args else message))

    def critical(self, message: str, *args: Any, **_kwargs: Any) -> None:
        self.messages.append(("critical", message % args if args else message))


@pytest.fixture
def secure_config(monkeypatch: pytest.MonkeyPatch) -> security.SecureConfig:
    monkeypatch.setenv("EXECUTION_MODE", "paper")
    monkeypatch.delenv("APP_ENV", raising=False)
    cfg = security.SecureConfig(master_key="phase2-test-key")
    cfg.audit_logger = cast(Any, _FakeAuditLogger())
    return cfg


def test_secure_config_encrypts_decrypts_and_detects_encrypted_values(
    secure_config: security.SecureConfig,
) -> None:
    encrypted = secure_config.encrypt_value("super-secret")

    assert encrypted != "super-secret"
    assert secure_config._is_encrypted(encrypted) is True
    assert secure_config.decrypt_value(encrypted) == "super-secret"
    assert secure_config.encrypt_value("") == ""
    assert secure_config.decrypt_value("") == ""
    assert secure_config._is_encrypted("short") is False


def test_secure_config_masks_nested_payloads_and_strings(
    secure_config: security.SecureConfig,
) -> None:
    payload = {
        "api_key": "KEY-1234567890SECRET",
        "nested": ["plain", "TOKENabcdefghijklmnopqrstuvwxyz"],
        "number": 5,
    }

    masked = secure_config.mask_sensitive_data(payload)

    assert masked["api_key"].startswith("KEY-")
    assert "***" in masked["api_key"]
    assert masked["nested"][0] == "plain"
    assert masked["nested"][1].startswith("TOKE")
    assert masked["number"] == 5
    assert secure_config.mask_sensitive_data("abc") == "***"


def test_get_and_set_secure_config_cache_and_audit(
    secure_config: security.SecureConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_calls: list[str] = []

    def fake_get_env(key: str, default: Any = None, **_kwargs: Any) -> Any:
        env_calls.append(key)
        return {"NEWS_API_KEY": "news-key-value"}.get(key, default)

    monkeypatch.setattr(security, "get_env", fake_get_env)

    assert secure_config.get_secure_config("NEWS_API_KEY") == "news-key-value"
    assert secure_config.get_secure_config("NEWS_API_KEY") == "news-key-value"
    secure_config.set_secure_config("CUSTOM", "value", encrypt=False)
    assert secure_config.get_secure_config("CUSTOM") == "value"
    secure_config.log_audit_event(
        security.AuditEventType.AUTHENTICATION,
        "login",
        "operator",
        severity=security.SecurityLevel.ERROR,
        details={"password": "PASSWORDabcdefghijklmnopqrstuvwxyz"},
        result="failure",
    )

    assert env_calls == ["NEWS_API_KEY"]
    assert len(cast(Any, secure_config.audit_logger).records) >= 4


def test_safe_logger_cleans_messages(secure_config: security.SecureConfig) -> None:
    base_logger = _FakeLogger()
    safe = security.SafeLogger(cast(Any, base_logger), secure_config)

    safe.info("using TOKENabcdefghijklmnopqrstuvwxyz for request")
    safe.error("key is SECRETabcdefghijklmnopqrstuvwxyz")

    assert base_logger.messages[0][0] == "info"
    assert "***" in base_logger.messages[0][1]
    assert "***" in base_logger.messages[1][1]


def test_security_manager_api_keys_validation_health_and_masking(
    secure_config: security.SecureConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = security.SecurityManager.__new__(security.SecurityManager)
    cast(Any, manager).logger = _FakeLogger()
    manager.secure_config = secure_config
    manager.safe_logger = security.SafeLogger(cast(Any, manager.logger), secure_config)
    manager._security_events = [
        security.AuditEvent(
            datetime.now(UTC),
            security.AuditEventType.SYSTEM_CHANGE,
            security.SecurityLevel.INFO,
            "system",
            "change",
            "config",
            {},
        ),
        security.AuditEvent(
            datetime.now(UTC) - timedelta(days=2),
            security.AuditEventType.SYSTEM_CHANGE,
            security.SecurityLevel.INFO,
            "system",
            "old",
            "config",
            {},
        ),
    ]
    manager._failed_access_attempts = 11
    manager._last_security_check = datetime.now(UTC)
    secure_config._config_cache["ALPACA_API_KEY"] = "ALPACA-KEY-1234567890"

    monkeypatch.setattr(security, "_CRYPTOGRAPHY_AVAILABLE", False)

    assert manager.get_api_key("alpaca") == "ALPACA-KEY-1234567890"
    assert manager.get_api_key("unknown") is None
    assert manager.validate_api_key("alpaca", "short") is False
    assert manager.validate_api_key("alpaca", "ALPACAKEY123456789012") is True
    assert manager.validate_api_key("news", "n" * 32) is True
    assert manager.validate_api_key("custom", "custom-key-1234567890") is True
    assert "***" in manager.mask_sensitive_data({"token": "TOKENabcdefghijklmnopqrstuvwxyz"})["token"]

    health = manager.check_security_health()

    assert health["overall_health"] == "degraded"
    assert health["recent_security_events"] == 1
    assert "Encryption library not available" in health["critical_issues"]
    assert "High number of failed access attempts" in health["critical_issues"]


def test_security_manager_rotate_key_and_module_helpers(
    secure_config: security.SecureConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = security.SecurityManager.__new__(security.SecurityManager)
    cast(Any, manager).logger = _FakeLogger()
    manager.secure_config = secure_config
    manager.safe_logger = security.SafeLogger(cast(Any, manager.logger), secure_config)
    manager._security_events = []
    manager._failed_access_attempts = 0
    manager._last_security_check = datetime.now(UTC)
    secure_config.set_secure_config("NEWS_API_KEY", "news-secret")

    monkeypatch.setattr(security, "get_security_manager", lambda: manager)
    monkeypatch.setattr(manager, "_generate_new_key", lambda: "rotated-key")

    assert manager.rotate_encryption_key() is True
    assert manager.get_api_key("news") == "news-secret"
    assert security.get_safe_logger("unit").secure_config is manager.secure_config
    assert "***" in security.mask_sensitive_data({"key": "KEYabcdefghijklmnopqrstuvwxyz"})["key"]
    security.log_security_event(
        security.AuditEventType.DATA_ACCESS,
        "read",
        "dataset",
        details={"token": "TOKENabcdefghijklmnopqrstuvwxyz"},
    )
