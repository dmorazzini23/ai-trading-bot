"""Tests for unified Alpaca credential handling across execution and data."""

from __future__ import annotations

import importlib
import logging
import sys
from typing import Iterator

import pytest

from ai_trading.utils import env as env_utils


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
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






def test_managed_secrets_backend_defers_missing_creds_downgrade_log(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """AWS-managed secrets hydrate after import, so startup should not emit a false downgrade."""

    monkeypatch.setenv("AI_TRADING_SECRETS_BACKEND", "aws-secrets-manager")

    caplog.set_level(logging.INFO)
    importlib.import_module("ai_trading.data.fetch")

    downgrade_logs = [record for record in caplog.records if record.message == "DATA_PROVIDER_DOWNGRADED"]
    assert not downgrade_logs
    assert env_utils.is_data_feed_downgraded()


def test_resolve_feed_cache_tracks_env_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing credentials should update the cached feed override automatically."""

    # Prime the cache with missing credentials to emulate prior downgrade decisions.
    assert env_utils.resolve_alpaca_feed(None) is None

    monkeypatch.setenv("ALPACA_API_KEY", "cache-fix-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "cache-fix-secret")

    # The next lookup should observe the new credentials without an explicit refresh.
    assert env_utils.resolve_alpaca_feed(None) == "iex"


def test_reference_feed_falls_back_to_delayed_sip_without_sip_entitlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALPACA_REFERENCE_FEED", "sip")
    monkeypatch.setenv("ALPACA_ALLOW_SIP", "0")
    monkeypatch.setenv("ALPACA_HAS_SIP", "0")
    monkeypatch.setenv("ALPACA_SIP_ENTITLED", "0")
    assert env_utils.resolve_alpaca_reference_feed(None) == "delayed_sip"
