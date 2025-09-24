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
        "ALPACA_DATA_API_KEY",
        "ALPACA_DATA_SECRET_KEY",
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    env_utils.refresh_alpaca_credentials_cache()
    # Remove cached data.fetch module to allow import-time hooks to rerun.
    sys.modules.pop("ai_trading.data.fetch", None)
    yield
    env_utils.refresh_alpaca_credentials_cache()
    sys.modules.pop("ai_trading.data.fetch", None)


def test_execution_only_creds_downgrade_data_feed(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Execution credentials without data keys should downgrade the data feed."""

    monkeypatch.setenv("APCA_API_KEY_ID", "exec-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "exec-secret")

    creds = env_utils.get_resolved_alpaca_credentials()
    assert creds.api_key == "exec-key"
    assert creds.secret_key == "exec-secret"
    assert creds.api_source == "APCA_API_KEY_ID"
    assert creds.secret_source == "APCA_API_SECRET_KEY"
    assert creds.has_execution_credentials()
    assert not creds.has_data_credentials()

    caplog.set_level(logging.INFO)
    data_fetch = importlib.import_module("ai_trading.data.fetch")

    downgrade_logs = [record for record in caplog.records if record.message == "DATA_PROVIDER_DOWNGRADED"]
    assert len(downgrade_logs) == 1
    record = downgrade_logs[0]
    assert getattr(record, "from") == "alpaca_iex"
    assert getattr(record, "to") == "yahoo"
    assert getattr(record, "reason") == "missing_data_keys"

    assert env_utils.is_data_feed_downgraded()
    assert env_utils.get_data_feed_override() == "yahoo"
    assert env_utils.get_data_feed_downgrade_reason() == "missing_data_keys"
    assert env_utils.resolve_alpaca_feed(None) is None

    # The data fetcher should also treat Alpaca credentials as unavailable in this mode.
    assert data_fetch._has_alpaca_keys() is False


def test_canonical_creds_enable_alpaca(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Canonical Alpaca keys should keep Alpaca as the primary data feed."""

    monkeypatch.setenv("ALPACA_API_KEY", "real-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "real-secret")

    creds = env_utils.get_resolved_alpaca_credentials()
    assert creds.api_source == "ALPACA_API_KEY"
    assert creds.secret_source == "ALPACA_SECRET_KEY"
    assert creds.has_data_credentials()

    caplog.set_level(logging.INFO)
    data_fetch = importlib.import_module("ai_trading.data.fetch")

    assert env_utils.is_data_feed_downgraded() is False
    assert env_utils.get_data_feed_override() is None
    assert env_utils.resolve_alpaca_feed(None) == "iex"
    assert data_fetch._has_alpaca_keys() is True

    downgrade_logs = [record for record in caplog.records if record.message == "DATA_PROVIDER_DOWNGRADED"]
    assert not downgrade_logs
