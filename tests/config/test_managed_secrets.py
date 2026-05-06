from __future__ import annotations

import os

import pytest

from ai_trading.config import managed_secrets
from ai_trading.config.management import (
    clear_runtime_env_overrides,
    get_env,
)
from ai_trading.utils import env as env_utils


def test_hydrate_managed_secrets_noops_when_backend_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_TRADING_SECRETS_BACKEND", "none")

    summary = managed_secrets.hydrate_managed_secrets(required_keys=("ALPACA_API_KEY",))

    assert summary["hydrated_count"] == 0


def test_hydrate_managed_secrets_sets_runtime_overrides_without_os_environ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_runtime_env_overrides(("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "WEBHOOK_SECRET"))
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.setenv("AI_TRADING_SECRETS_BACKEND", "aws-secrets-manager")
    monkeypatch.setenv("AI_TRADING_AWS_SECRET_ID", "ai-trading-bot/prod")
    monkeypatch.setenv("AI_TRADING_REQUIRE_MANAGED_SECRETS", "1")
    monkeypatch.setattr(
        managed_secrets,
        "_fetch_aws_secret_payload",
        lambda secret_id, *, region, profile: {
            "ALPACA_API_KEY": "remote-key",
            "ALPACA_SECRET_KEY": "remote-secret",
            "WEBHOOK_SECRET": "remote-webhook",
        },
    )

    summary = managed_secrets.hydrate_managed_secrets(
        required_keys=("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "WEBHOOK_SECRET")
    )

    assert summary["hydrated_count"] == 3
    assert get_env("ALPACA_API_KEY", "", resolve_aliases=False) == "remote-key"
    assert "ALPACA_API_KEY" not in os.environ
    assert env_utils.resolve_alpaca_feed(None) == "iex"

    clear_runtime_env_overrides(("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "WEBHOOK_SECRET"))
    env_utils.refresh_alpaca_credentials_cache()


def test_hydrate_managed_secrets_fails_when_required_payload_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_runtime_env_overrides(("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "WEBHOOK_SECRET"))
    monkeypatch.setenv("AI_TRADING_SECRETS_BACKEND", "aws-secrets-manager")
    monkeypatch.setenv("AI_TRADING_AWS_SECRET_ID", "ai-trading-bot/prod")
    monkeypatch.setenv("AI_TRADING_REQUIRE_MANAGED_SECRETS", "1")
    monkeypatch.setattr(
        managed_secrets,
        "_fetch_aws_secret_payload",
        lambda secret_id, *, region, profile: {"ALPACA_API_KEY": "remote-key"},
    )

    with pytest.raises(RuntimeError, match="ALPACA_SECRET_KEY"):
        managed_secrets.hydrate_managed_secrets(
            required_keys=("ALPACA_API_KEY", "ALPACA_SECRET_KEY")
        )
