from __future__ import annotations

import base64

import pytest

import ai_trading.security as security


class _BrokenFernet:
    def encrypt(self, _data: bytes) -> bytes:
        raise ValueError("boom")

    def decrypt(self, _token: bytes) -> bytes:
        raise ValueError("boom")


def test_secure_config_requires_master_key_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("MASTER_ENCRYPTION_KEY", raising=False)

    with pytest.raises(RuntimeError, match="MASTER_ENCRYPTION_KEY"):
        security.SecureConfig()


def test_secure_config_requires_cryptography_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.setenv("MASTER_ENCRYPTION_KEY", "test-master-key")
    monkeypatch.setattr(security, "_CRYPTOGRAPHY_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="Cryptography library is required"):
        security.SecureConfig()


def test_encrypt_decrypt_fail_closed_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.setenv("MASTER_ENCRYPTION_KEY", "test-master-key")

    cfg = security.SecureConfig()
    cfg._fernet = _BrokenFernet()

    with pytest.raises(RuntimeError, match="Encryption failed"):
        cfg.encrypt_value("abc")

    encoded = base64.urlsafe_b64encode(b"abc").decode()
    with pytest.raises(RuntimeError, match="Decryption failed"):
        cfg.decrypt_value(encoded)


def test_encrypt_decrypt_allow_fallback_outside_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "paper")
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("MASTER_ENCRYPTION_KEY", raising=False)

    cfg = security.SecureConfig(master_key="unit-test-key")
    cfg._fernet = _BrokenFernet()

    assert cfg.encrypt_value("abc") == "abc"
    encoded = base64.urlsafe_b64encode(b"abc").decode()
    assert cfg.decrypt_value(encoded) == encoded
