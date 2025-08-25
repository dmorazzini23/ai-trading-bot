from ai_trading.config.management import TradingConfig


def test_from_env_accepts_extras_and_scales():
    env = {
        "CAPITAL_CAP": "0.05",
        "DATA_FEED": "iex",
        "DATA_PROVIDER": "alpaca",
        "UNRELATED": "OK",
    }
    cfg = TradingConfig.from_env(env)
    assert cfg.capital_cap == 0.05
    assert cfg.extras["UNRELATED"] == "OK"
    assert cfg.derive_cap_from_settings(100000.0, fallback=8000.0) == 5000.0

