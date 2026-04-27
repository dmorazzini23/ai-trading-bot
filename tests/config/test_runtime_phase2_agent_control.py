from __future__ import annotations

import pytest

import ai_trading.config.runtime as runtime_config


def test_env_snapshot_applies_operator_friendly_alias_overrides(monkeypatch) -> None:
    monkeypatch.setattr(runtime_config, "_managed_env_snapshot", lambda: {})

    snap = runtime_config._env_snapshot(
        {
            "DATA_PROVIDER": "yahoo",
            "PAPER": "1",
            "AI_TRADING_TRADING_MODE": "conservative",
        }
    )

    assert snap["DATA_PROVIDER_PRIORITY"] == "yahoo"
    assert snap["EXECUTION_MODE"] == "paper"
    assert snap["AI_TRADING_TRADING_MODE"] == "conservative"
    assert snap.override_keys == frozenset(
        {"DATA_PROVIDER", "PAPER", "AI_TRADING_TRADING_MODE"}
    )


def test_mode_resolution_strict_mode_prefers_runtime_override() -> None:
    mode, source, precedence = runtime_config._resolve_mode_selection(
        {
            "AI_TRADING_TRADING_MODE": "aggressive",
            "__TRADING_MODE_BASE__": "conservative",
            "__TRADING_MODE_BASE_SOURCE__": "AI_TRADING_TRADING_MODE",
            "__TRADING_MODE_OVERRIDE__": "balanced",
            "__TRADING_MODE_OVERRIDE_SOURCE__": "operator",
            "AI_TRADING_TRADING_MODE_PRECEDENCE": "strict_mode",
        }
    )

    assert (mode, source, precedence) == ("balanced", "operator", "strict_mode")


def test_mode_resolution_env_wins_prefers_base_mode() -> None:
    mode, source, precedence = runtime_config._resolve_mode_selection(
        {
            "AI_TRADING_TRADING_MODE": "aggressive",
            "__TRADING_MODE_BASE__": "conservative",
            "__TRADING_MODE_BASE_SOURCE__": "AI_TRADING_TRADING_MODE",
            "__TRADING_MODE_OVERRIDE__": "balanced",
            "__TRADING_MODE_OVERRIDE_SOURCE__": "operator",
            "AI_TRADING_TRADING_MODE_PRECEDENCE": "env_wins",
        }
    )

    assert (mode, source, precedence) == (
        "conservative",
        "AI_TRADING_TRADING_MODE",
        "env_wins",
    )


def test_from_env_rejects_fail_fast_deprecated_position_size(monkeypatch) -> None:
    monkeypatch.setattr(runtime_config, "_managed_env_snapshot", lambda: {})

    with pytest.raises(RuntimeError, match="MAX_POSITION_SIZE is deprecated"):
        runtime_config.TradingConfig.from_env(
            {
                "MAX_POSITION_SIZE": "100",
                "MAX_DRAWDOWN_THRESHOLD": "0.1",
            },
            allow_missing_drawdown=True,
        )
