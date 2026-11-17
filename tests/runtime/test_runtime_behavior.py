import asyncio
from threading import Event
from unittest.mock import patch

from ai_trading.main import _get_int_env, start_api_with_signal


def test_get_int_env_invalid_returns_default(monkeypatch):
    """Invalid integer env vars should fall back to the provided default."""
    monkeypatch.setenv("TEST_INT", "not-an-int")
    assert _get_int_env("TEST_INT", 5) == 5


def test_start_api_with_signal_sets_error_on_failure():
    """If start_api raises, the error flag should be set."""
    api_ready = Event()
    api_error = Event()
    with patch("ai_trading.main.start_api", side_effect=RuntimeError):
        start_api_with_signal(api_ready, api_error)
    assert api_error.is_set()


