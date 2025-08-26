import pytest

from ai_trading.config import management as config
from ai_trading.config import settings as settings_mod


def test_get_env_entrypoint(monkeypatch):
    """Canonical get_env returns casted values and handles required flags."""
    monkeypatch.setenv("FOO", "123")
    assert config.get_env("FOO") == "123"
    assert config.get_env("FOO", cast=int) == 123
    assert config.get_env("BAR", "baz") == "baz"
    with pytest.raises(RuntimeError):
        config.get_env("MISSING", required=True)


def test_get_settings_singleton():
    """Management and settings helpers share the same Settings instance."""
    assert config.get_settings() is settings_mod.get_settings()
