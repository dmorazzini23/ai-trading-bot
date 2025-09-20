"""Tests for ai_trading.tools.env_validate."""

from __future__ import annotations

import importlib.util

import pytest

from ai_trading.tools.env_validate import validate_env


@pytest.fixture(autouse=True)
def mock_required_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure required package checks pass during tests."""

    monkeypatch.setattr(importlib.util, 'find_spec', lambda pkg: object())


def _base_env() -> dict[str, str]:
    return {
        'ALPACA_API_KEY': 'key',
        'ALPACA_SECRET_KEY': 'secret',
    }


def test_validate_env_accepts_base_url_only() -> None:
    env = {
        **_base_env(),
        'ALPACA_BASE_URL': 'https://paper-api.alpaca.markets',
    }

    assert validate_env(env) == []


def test_validate_env_accepts_api_url_only() -> None:
    env = {
        **_base_env(),
        'ALPACA_API_URL': 'https://paper-api.alpaca.markets',
    }

    assert validate_env(env) == []


def test_validate_env_requires_some_url() -> None:
    env = _base_env()

    missing = validate_env(env)

    assert 'ALPACA_BASE_URL' in missing
