"""Tests for unified Alpaca credential handling across execution and data."""

from __future__ import annotations

import importlib
import logging
import sys

import pytest

from ai_trading.utils import env as env_utils


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure environment-related caches are reset between tests."""

    for key in (
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    env_utils.refresh_alpaca_credentials_cache()
    # Remove cached data.fetch module to allow import-time hooks to rerun.
    sys.modules.pop("ai_trading.data.fetch", None)
    yield
    env_utils.refresh_alpaca_credentials_cache()
    sys.modules.pop("ai_trading.data.fetch", None)


def test_missing_creds_downgrade_data_feed(caplog: pytest.LogCaptureFixture) -> None:
    """Missing canonical credentials should disable Alpaca data access."""

    creds = env_utils.get_resolved_alpaca_credentials()
    assert creds.api_key is None
    assert creds.secret_key is None
    assert not creds.has_execution_credentials()
    assert not creds.has_data_credentials()

    caplog.set_level(logging.INFO)
    data_fetch = importlib.import_module("ai_trading.data.fetch")

    downgrade_logs = [record for record in caplog.records if record.message == "DATA_PROVIDER_DOWNGRADED"]
    assert len(downgrade_logs) == 1
    record = downgrade_logs[0]
    assert getattr(record, "from") == "alpaca_iex"
    assert getattr(record, "to") == "yahoo"
    assert getattr(record, "reason") == "missing_credentials"

    assert env_utils.is_data_feed_downgraded()
    assert env_utils.get_data_feed_override() == "yahoo"
    assert env_utils.get_data_feed_downgrade_reason() == "missing_credentials"
    assert env_utils.resolve_alpaca_feed(None) is None

    # The data fetcher should also treat Alpaca credentials as unavailable in this mode.
    assert data_fetch._has_alpaca_keys() is False


def test_canonical_creds_enable_alpaca(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Canonical Alpaca keys should keep Alpaca as the primary data feed."""

    monkeypatch.setenv("ALPACA_API_KEY", "real-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "real-secret")

    creds = env_utils.get_resolved_alpaca_credentials()
    assert creds.has_data_credentials()

    caplog.set_level(logging.INFO)
    data_fetch = importlib.import_module("ai_trading.data.fetch")

    assert env_utils.is_data_feed_downgraded() is False
    assert env_utils.get_data_feed_override() is None
    assert env_utils.resolve_alpaca_feed(None) == "iex"
    assert data_fetch._has_alpaca_keys() is True

    downgrade_logs = [record for record in caplog.records if record.message == "DATA_PROVIDER_DOWNGRADED"]
    assert not downgrade_logs


def test_resolve_feed_cache_tracks_env_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing credentials should update the cached feed override automatically."""

    # Prime the cache with missing credentials to emulate prior downgrade decisions.
    assert env_utils.resolve_alpaca_feed(None) is None

    monkeypatch.setenv("ALPACA_API_KEY", "cache-fix-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "cache-fix-secret")

    # The next lookup should observe the new credentials without an explicit refresh.
    assert env_utils.resolve_alpaca_feed(None) == "iex"
