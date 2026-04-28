from ai_trading.config import scaling


def test_from_env_accepts_extras_and_scales():
    env = {
        "AI_TRADING_CAPITAL_CAP": "0.05",
        "DATA_FEED": "iex",
        "DATA_PROVIDER": "alpaca",
        "UNRELATED": "OK",
        "NUMERIC": "2",
        "AI_TRADING_SCALING_EXTRA_NUMERIC": "2",
    }
    cfg = scaling.from_env(env)
    assert cfg.capital_cap == 0.05
    assert "UNRELATED" not in cfg.extras
    assert "NUMERIC" not in cfg.extras
    assert cfg.extras["AI_TRADING_SCALING_EXTRA_NUMERIC"] == 2
    assert cfg.derive_cap_from_settings(100000.0, fallback=8000.0) == 5000.0


def test_from_env_initializes_empty_extras():
    cfg = scaling.from_env({"AI_TRADING_CAPITAL_CAP": "0.04"})
    assert cfg.extras == {}


def test_from_env_does_not_copy_unknown_secrets_to_extras():
    cfg = scaling.from_env(
        {
            "AI_TRADING_CAPITAL_CAP": "0.04",
            "RANDOM_SECRET_TOKEN": "super-secret",
            "TRADING_CONFIG_EXTRAS": '{"safe": "ok", "api_key": "secret"}',
        }
    )

    assert "RANDOM_SECRET_TOKEN" not in cfg.extras
    assert cfg.extras["safe"] == "ok"
    assert cfg.extras["api_key"] == "***"
