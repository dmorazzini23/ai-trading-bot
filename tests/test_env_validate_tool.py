"""Tests for ai_trading.tools.env_validate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from ai_trading.tools import env_validate as env_validate_tool
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
        'ALPACA_TRADING_BASE_URL': 'https://paper-api.alpaca.markets',
    }

    assert validate_env(env) == []


def test_validate_env_rejects_deprecated_api_url() -> None:
    env = {
        **_base_env(),
        'ALPACA_API_URL': 'https://paper-api.alpaca.markets',
    }

    missing = validate_env(env)
    assert any('ALPACA_API_URL is deprecated' in entry for entry in missing)
    assert 'ALPACA_TRADING_BASE_URL' in missing


def test_validate_env_requires_some_url() -> None:
    env = _base_env()

    missing = validate_env(env)

    assert 'ALPACA_TRADING_BASE_URL' in missing


def test_main_loads_dotenv_from_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_path = tmp_path / '.env'
    env_path.write_text(
        '\n'.join(
            (
                'ALPACA_API_KEY=key',
                'ALPACA_SECRET_KEY=secret',
                'ALPACA_TRADING_BASE_URL=https://paper-api.alpaca.markets',
            )
        ),
        encoding='utf-8',
    )
    monkeypatch.chdir(tmp_path)
    for key in ('ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'ALPACA_TRADING_BASE_URL'):
        monkeypatch.delenv(key, raising=False)

    assert env_validate_tool.main([]) == 0
