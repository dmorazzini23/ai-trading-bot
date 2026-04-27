from __future__ import annotations

import logging

import pytest

import ai_trading.config.runtime as runtime_config


def test_validate_override_keys_rejects_unknown() -> None:
    with pytest.raises(KeyError, match="UNKNOWN_SETTING"):
        runtime_config._validate_override_keys({"UNKNOWN_SETTING": "1"})


def test_reject_legacy_apca_env_lists_preview(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_config,
        "_managed_env_snapshot",
        lambda: {
            "APCA_API_KEY_ID": "old",
            "APCA_API_SECRET_KEY": "secret",
            "ALPACA_API_KEY": "new",
        },
    )

    with pytest.raises(RuntimeError) as exc:
        runtime_config._reject_legacy_apca_env()

    assert "APCA_* environment variables" in str(exc.value)
    assert "APCA_API_KEY_ID" in str(exc.value)


def test_detect_env_alias_conflict_warns_outside_live(caplog) -> None:
    spec = runtime_config.SPEC_BY_FIELD["webhook_secret"]
    env_map = {"WEBHOOK_SECRET": "canonical", "AI_TRADING_WEBHOOK_SECRET": "alias"}

    with caplog.at_level(logging.WARNING, logger=runtime_config.logger.name):
        runtime_config._detect_env_alias_conflict(spec, env_map)

    assert any(record.message == "CONFIG_ENV_ALIAS_CONFLICT" for record in caplog.records)


def test_detect_env_alias_conflict_fails_in_live() -> None:
    spec = runtime_config.SPEC_BY_FIELD["webhook_secret"]
    env_map = {
        "EXECUTION_MODE": "live",
        "WEBHOOK_SECRET": "canonical",
        "AI_TRADING_WEBHOOK_SECRET": "alias",
    }

    with pytest.raises(RuntimeError, match="Conflicting values"):
        runtime_config._detect_env_alias_conflict(spec, env_map)


def test_trading_config_update_validates_unknown_and_recomputes_paper() -> None:
    cfg = runtime_config.TradingConfig.from_env(
        {
            "APP_ENV": "prod",
            "ALPACA_TRADING_BASE_URL": "https://api.alpaca.markets",
            "MAX_DRAWDOWN_THRESHOLD": "0.1",
        },
        allow_missing_drawdown=True,
    )

    with pytest.raises(AttributeError, match="no fields"):
        cfg.update(nope=True)

    assert cfg.paper is False
    assert cfg.update(
        app_env="test",
        alpaca_base_url="https://paper-api.alpaca.markets",
    ) is cfg
    assert cfg.paper is True
