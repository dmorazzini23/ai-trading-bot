from __future__ import annotations

import pytest

from ai_trading.config.management import TradingConfig


def _base_env() -> dict[str, str]:
    return {
        "MAX_DRAWDOWN_THRESHOLD": "0.15",
        "MAX_POSITION_MODE": "STATIC",
    }


def test_alias_conflict_warns_and_prefers_canonical_in_non_live() -> None:
    env = _base_env() | {
        "APP_ENV": "test",
        "EXECUTION_MODE": "sim",
        "MAX_POSITION_SIZE": "7000",
        "AI_TRADING_MAX_POSITION_SIZE": "9000",
    }
    with pytest.raises(RuntimeError, match="MAX_POSITION_SIZE is deprecated"):
        TradingConfig.from_env(env)


def test_alias_conflict_raises_in_live_mode() -> None:
    env = _base_env() | {
        "APP_ENV": "production",
        "EXECUTION_MODE": "live",
        "AI_TRADING_MAX_POSITION_SIZE": "9000",
    }

    with pytest.raises(RuntimeError, match="AI_TRADING_MAX_POSITION_SIZE is deprecated"):
        TradingConfig.from_env(env)


def test_alias_matching_values_allowed_in_live_mode() -> None:
    env = _base_env() | {
        "APP_ENV": "production",
        "EXECUTION_MODE": "live",
        "AI_TRADING_SIGNAL_MAX_POSITION_SIZE": "7000",
    }

    cfg = TradingConfig.from_env(env)
    assert cfg.max_position_size == pytest.approx(7000.0)
