import os

import pytest


def test_empty_env_uses_default(monkeypatch):
    # Ensure empty env value falls back to field default via validator
    monkeypatch.setenv("AI_TRADING_MODEL_PATH", "")
    from ai_trading.settings import Settings

    s = Settings()
    assert s.model_path == "trained_model.pkl"


@pytest.mark.parametrize(
    "env_var, expected",
    [
        ("AI_TRADING_HALT_FLAG_PATH", "halt.flag"),
        ("AI_TRADING_RL_MODEL_PATH", "models/runtime/rl_agent.zip"),
    ],
)
def test_empty_env_paths_use_defaults(monkeypatch, env_var, expected):
    monkeypatch.setenv(env_var, "")
    from ai_trading.settings import Settings

    s = Settings()
    # Map env var to attribute name
    attr = {
        "AI_TRADING_HALT_FLAG_PATH": "halt_flag_path",
        "AI_TRADING_RL_MODEL_PATH": "rl_model_path",
    }[env_var]
    assert getattr(s, attr) == expected


def test_cache_ttls_default(monkeypatch):
    monkeypatch.delenv("MINUTE_CACHE_TTL", raising=False)
    monkeypatch.delenv("MARKET_CACHE_TTL", raising=False)
    from ai_trading.settings import Settings

    s = Settings()
    assert s.minute_cache_ttl == 300
    assert s.market_cache_ttl == 86400
