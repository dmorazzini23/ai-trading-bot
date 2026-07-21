from __future__ import annotations

import pytest

from ai_trading.config.runtime import TradingConfig


def _paper_sampling_env(**updates: str) -> dict[str, str]:
    env = {
        "APP_ENV": "test",
        "EXECUTION_MODE": "paper",
        "ALPACA_TRADING_BASE_URL": "https://paper-api.alpaca.markets",
        "AI_TRADING_LAUNCH_PROFILE": "paper_trade",
        "AI_TRADING_PAPER_SAMPLING_ENABLED": "1",
        "AI_TRADING_PAPER_SAMPLING_ALLOWED_SYMBOLS": "AAPL,AMZN,MSFT",
        "AI_TRADING_PAPER_SAMPLING_MAX_NOTIONAL_PER_ORDER": "750",
        "AI_TRADING_PAPER_SAMPLING_PASSIVE_ONLY": "1",
        "AI_TRADING_PAPER_SAMPLING_RELAX_EDGE_GATES_ENABLED": "0",
        "AI_TRADING_PAPER_SAMPLING_PASSIVE_REPRICE_ENABLED": "1",
        "MAX_DRAWDOWN_THRESHOLD": "0.2",
    }
    env.update(updates)
    return env


def test_passive_reprice_defaults_are_conservative_when_disabled() -> None:
    cfg = TradingConfig.from_env(
        {
            "APP_ENV": "test",
            "EXECUTION_MODE": "sim",
            "AI_TRADING_PAPER_SAMPLING_ENABLED": "0",
            "AI_TRADING_PAPER_SAMPLING_PASSIVE_REPRICE_ENABLED": "0",
            "MAX_DRAWDOWN_THRESHOLD": "0.2",
        }
    )

    assert cfg.paper_sampling_passive_reprice_enabled is False
    assert cfg.paper_sampling_passive_reprice_timeout_sec == 45.0
    assert cfg.paper_sampling_passive_reprice_max_retries == 2
    assert cfg.paper_sampling_passive_reprice_cooldown_sec == 30.0
    assert cfg.paper_sampling_passive_reprice_quote_max_age_ms == 2500.0
    assert cfg.paper_sampling_passive_reprice_max_spread_bps == 20.0
    assert cfg.paper_sampling_passive_reprice_max_actions_per_cycle == 2
    assert cfg.paper_sampling_passive_reprice_hard_cancel_before_close_sec == 300.0


def test_passive_reprice_can_be_enabled_for_governed_paper_sampling() -> None:
    cfg = TradingConfig.from_env(_paper_sampling_env())

    assert cfg.paper_sampling_passive_reprice_enabled is True
    assert cfg.execution_mode == "paper"
    assert cfg.paper is True


def test_passive_reprice_fails_closed_when_paper_sampling_is_disabled() -> None:
    cfg = TradingConfig.from_env(
        {
            "APP_ENV": "test",
            "EXECUTION_MODE": "sim",
            "AI_TRADING_PAPER_SAMPLING_ENABLED": "0",
            "AI_TRADING_PAPER_SAMPLING_PASSIVE_REPRICE_ENABLED": "1",
            "MAX_DRAWDOWN_THRESHOLD": "0.2",
        }
    )

    assert cfg.paper_sampling_enabled is False
    assert cfg.paper_sampling_passive_reprice_enabled is False


@pytest.mark.parametrize(
    ("env_key", "env_value", "field"),
    (
        (
            "AI_TRADING_PAPER_SAMPLING_PASSIVE_REPRICE_TIMEOUT_SEC",
            "7.99",
            "paper_sampling_passive_reprice_timeout_sec",
        ),
        (
            "AI_TRADING_PAPER_SAMPLING_PASSIVE_REPRICE_TIMEOUT_SEC",
            "3601",
            "paper_sampling_passive_reprice_timeout_sec",
        ),
        (
            "AI_TRADING_PAPER_SAMPLING_PASSIVE_REPRICE_MAX_RETRIES",
            "-1",
            "paper_sampling_passive_reprice_max_retries",
        ),
        (
            "AI_TRADING_PAPER_SAMPLING_PASSIVE_REPRICE_MAX_RETRIES",
            "9",
            "paper_sampling_passive_reprice_max_retries",
        ),
        (
            "AI_TRADING_PAPER_SAMPLING_PASSIVE_REPRICE_COOLDOWN_SEC",
            "-0.1",
            "paper_sampling_passive_reprice_cooldown_sec",
        ),
        (
            "AI_TRADING_PAPER_SAMPLING_PASSIVE_REPRICE_COOLDOWN_SEC",
            "1801",
            "paper_sampling_passive_reprice_cooldown_sec",
        ),
    ),
)
def test_passive_reprice_rejects_bounded_value_violations(
    env_key: str,
    env_value: str,
    field: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        TradingConfig.from_env(_paper_sampling_env(**{env_key: env_value}))


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        (
            {
                "APP_ENV": "prod",
                "EXECUTION_MODE": "live",
                "ALPACA_TRADING_BASE_URL": "https://api.alpaca.markets",
            },
            "PAPER_SAMPLING_ENABLED requires EXECUTION_MODE=paper",
        ),
        (
            {"ALPACA_TRADING_BASE_URL": "https://api.alpaca.markets"},
            "a paper Alpaca base URL",
        ),
        (
            {"AI_TRADING_LAUNCH_PROFILE": "live_canary"},
            "non-live launch profile",
        ),
        (
            {"AI_TRADING_PAPER_SAMPLING_PASSIVE_ONLY": "0"},
            "PASSIVE_ONLY must remain enabled",
        ),
        (
            {"AI_TRADING_PAPER_SAMPLING_RELAX_EDGE_GATES_ENABLED": "1"},
            "RELAX_EDGE_GATES_ENABLED must remain disabled",
        ),
        (
            {"AI_TRADING_PAPER_SAMPLING_MAX_NOTIONAL_PER_ORDER": "751"},
            "MAX_NOTIONAL_PER_ORDER must be <= 750",
        ),
    ),
)
def test_enabled_passive_reprice_inherits_sampling_safety_invariants(
    updates: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TradingConfig.from_env(_paper_sampling_env(**updates))
