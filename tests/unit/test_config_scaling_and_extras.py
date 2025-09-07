from ai_trading.config import scaling


def test_from_env_accepts_extras_and_scales():
    env = {
        "CAPITAL_CAP": "0.05",
        "DATA_FEED": "iex",
        "DATA_PROVIDER": "alpaca",
        "UNRELATED": "OK",
        "NUMERIC": "2",
    }
    cfg = scaling.from_env(env)
    assert cfg.capital_cap == 0.05
    assert cfg.extras["UNRELATED"] == "OK"
    assert cfg.extras["NUMERIC"] == 2
    assert cfg.derive_cap_from_settings(100000.0, fallback=8000.0) == 5000.0


def test_from_env_initializes_empty_extras():
    cfg = scaling.from_env({"CAPITAL_CAP": "0.04"})
    assert cfg.extras == {}
