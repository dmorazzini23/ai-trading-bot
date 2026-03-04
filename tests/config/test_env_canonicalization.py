from __future__ import annotations

import pytest

from ai_trading.config.management import canonical_env_map, validate_no_deprecated_env


def test_canonical_env_map_contains_required_pairs() -> None:
    mapping = canonical_env_map()

    assert mapping["AI_TRADING_SIGNAL_MAX_POSITION_SIZE"] == (
        "MAX_POSITION_SIZE",
        "AI_TRADING_MAX_POSITION_SIZE",
    )
    assert mapping["MAX_DRAWDOWN_THRESHOLD"] == ("AI_TRADING_MAX_DRAWDOWN_THRESHOLD",)
    assert mapping["TRADING__ALLOW_SHORTS"] == ("AI_TRADING_ALLOW_SHORT",)
    assert mapping["EXECUTION_ALLOW_FALLBACK_WITHOUT_NBBO"] == (
        "AI_TRADING_EXEC_ALLOW_FALLBACK_WITHOUT_NBBO",
    )
    assert mapping["SENTIMENT_API_KEY"] == ("NEWS_API_KEY",)
    assert mapping["ALPACA_TRADING_BASE_URL"] == ("ALPACA_API_URL", "ALPACA_BASE_URL")


def test_validate_no_deprecated_env_rejects_alpaca_aliases() -> None:
    env = {
        "ALPACA_TRADING_BASE_URL": "https://paper-api.alpaca.markets",
        "ALPACA_API_URL": "https://paper-api.alpaca.markets",
    }

    with pytest.raises(RuntimeError, match="ALPACA_API_URL"):
        validate_no_deprecated_env(env)


def test_validate_no_deprecated_env_rejects_sentiment_alias() -> None:
    env = {
        "SENTIMENT_API_KEY": "sentiment-key",
        "NEWS_API_KEY": "legacy-key",
    }

    with pytest.raises(RuntimeError, match="NEWS_API_KEY"):
        validate_no_deprecated_env(env)


def test_validate_no_deprecated_env_accepts_canonical_keys() -> None:
    env = {
        "ALPACA_TRADING_BASE_URL": "https://paper-api.alpaca.markets",
        "ALPACA_DATA_BASE_URL": "https://data.alpaca.markets",
        "AI_TRADING_SIGNAL_MAX_POSITION_SIZE": "1000",
        "MAX_DRAWDOWN_THRESHOLD": "0.1",
        "TRADING__ALLOW_SHORTS": "0",
        "EXECUTION_ALLOW_FALLBACK_WITHOUT_NBBO": "0",
        "SENTIMENT_API_KEY": "key",
    }

    validate_no_deprecated_env(env)
