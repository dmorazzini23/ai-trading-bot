from __future__ import annotations

import logging

import pytest

from ai_trading.config.management import TradingConfig


def _base_env() -> dict[str, str]:
    return {
        "MAX_DRAWDOWN_THRESHOLD": "0.15",
        "MAX_POSITION_MODE": "STATIC",
    }


def test_alias_conflict_warns_and_prefers_canonical_in_non_live(caplog: pytest.LogCaptureFixture) -> None:
    env = _base_env() | {
        "APP_ENV": "test",
        "EXECUTION_MODE": "sim",
        "MAX_POSITION_SIZE": "7000",
        "AI_TRADING_MAX_POSITION_SIZE": "9000",
    }
    caplog.set_level(logging.WARNING, logger="ai_trading.config.runtime")

    cfg = TradingConfig.from_env(env)

    assert cfg.max_position_size == pytest.approx(7000.0)
    conflict_logs = [record for record in caplog.records if record.msg == "CONFIG_ENV_ALIAS_CONFLICT"]
    assert conflict_logs


def test_alias_conflict_raises_in_live_mode() -> None:
    env = _base_env() | {
        "APP_ENV": "production",
        "EXECUTION_MODE": "live",
        "MAX_POSITION_SIZE": "7000",
        "AI_TRADING_MAX_POSITION_SIZE": "9000",
    }

    with pytest.raises(RuntimeError, match="Conflicting values for canonical/deprecated env keys"):
        TradingConfig.from_env(env)


def test_alias_matching_values_allowed_in_live_mode() -> None:
    env = _base_env() | {
        "APP_ENV": "production",
        "EXECUTION_MODE": "live",
        "MAX_POSITION_SIZE": "7000",
        "AI_TRADING_MAX_POSITION_SIZE": "07000",
    }

    cfg = TradingConfig.from_env(env)
    assert cfg.max_position_size == pytest.approx(7000.0)
