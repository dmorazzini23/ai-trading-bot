from __future__ import annotations

import os

import pytest

from ai_trading.config.management import (
    enforce_alpaca_feed_policy,
    get_trading_config,
    reload_trading_config,
)


@pytest.fixture(autouse=True)
def _reset_config(monkeypatch):
    """Ensure trading config reloads with test-specific environment."""

    # Snapshot original env values to restore after each test.
    original_env = {
        key: os.environ.get(key)
        for key in ("DATA_PROVIDER", "ALPACA_DATA_FEED", "DATA_FEED")
    }
    yield
    for key, value in original_env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    reload_trading_config()


def test_config_alpaca_feed_without_sip_falls_back(monkeypatch):
    monkeypatch.setenv("DATA_PROVIDER", "alpaca")
    monkeypatch.setenv("ALPACA_DATA_FEED", "iex")
    reload_trading_config()

    info = enforce_alpaca_feed_policy()

    assert info is not None
    assert info.get("status") == "fallback"
    assert info.get("fallback_provider") == "yahoo"

    cfg = get_trading_config()
    assert str(cfg.data_provider).lower() == "yahoo"
    assert tuple(cfg.data_provider_priority)[0] == "yahoo"

    assert os.environ.get("DATA_PROVIDER") == "yahoo"
    assert os.environ.get("DATA_PROVIDER_PRIORITY") == "yahoo"
    # Original feed setting remains unchanged
    assert os.environ.get("ALPACA_DATA_FEED") == "iex"


def test_config_alpaca_feed_defaults_to_sip(monkeypatch):
    monkeypatch.setenv("DATA_PROVIDER", "alpaca")
    monkeypatch.delenv("ALPACA_DATA_FEED", raising=False)
    monkeypatch.delenv("DATA_FEED", raising=False)
    reload_trading_config()

    info = enforce_alpaca_feed_policy()

    assert info == {"provider": "alpaca", "feed": "sip", "status": "sip"}
    assert os.environ.get("ALPACA_DATA_FEED") == "sip"
