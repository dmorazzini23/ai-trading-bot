"""Alpaca credential helpers expose a unified dataclass view."""

from __future__ import annotations

import builtins

import pytest

from ai_trading.broker import alpaca_credentials as creds_mod


def test_resolve_alpaca_credentials_defaults(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("ALPACA_BASE_URL", raising=False)
    creds = creds_mod.resolve_alpaca_credentials({})
    assert creds.api_key is None
    assert creds.secret_key is None
    assert creds.base_url.endswith("alpaca.markets")


def test_resolve_alpaca_credentials_from_mapping():
    data = {
        "ALPACA_API_KEY": "key",
        "ALPACA_SECRET_KEY": "secret",
        "ALPACA_BASE_URL": "https://live.alpaca.markets",
    }
    creds = creds_mod.resolve_alpaca_credentials(data)
    assert creds.api_key == "key"
    assert creds.secret_key == "secret"
    assert creds.base_url == "https://live.alpaca.markets"


def test_check_alpaca_available_handles_missing(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("alpaca"):
            raise ModuleNotFoundError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert creds_mod.check_alpaca_available() is False


def test_initialize_shadow_allows_missing_sdk(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "alpaca":
            raise ModuleNotFoundError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    stub = creds_mod.initialize({}, shadow=True)
    assert isinstance(stub, object)


def test_initialize_requires_sdk_when_not_shadow(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "alpaca":
            raise ModuleNotFoundError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError):
        creds_mod.initialize({}, shadow=False)
