from ai_trading.config.management import TradingConfig


def test_trading_config_env_aliases():
    env = {
        "DAILY_LOSS_LIMIT": "0.06",
        "CAPITAL_CAP": "0.05",
        "MAX_POSITION_MODE": "STATIC",
        "DATA_FEED": "iex",
        "DATA_PROVIDER": "alpaca",
        "PAPER": "true",
        "KELLY_FRACTION_MAX": "0.2",
        "MIN_SAMPLE_SIZE": "50",
        "CONFIDENCE_LEVEL": "0.9",
    }
    cfg = TradingConfig.from_env(env)
    assert cfg.dollar_risk_limit == 0.06
    assert cfg.capital_cap == 0.05
    assert cfg.max_position_mode == "STATIC"
    assert cfg.kelly_fraction_max == 0.2
    assert cfg.min_sample_size == 50
    assert cfg.confidence_level == 0.9
    snap = cfg.snapshot_sanitized()
    assert snap["data"]["feed"] == "iex"
    assert snap["data"]["provider"] == "alpaca"


def test_paper_inferred_from_base_url():
    env = {"ALPACA_BASE_URL": "https://paper-api.alpaca.markets"}
    cfg = TradingConfig.from_env(env)
    assert cfg.paper is True


def test_paper_false_when_live_prod():
    env = {
        "ALPACA_BASE_URL": "https://api.alpaca.markets",
        "APP_ENV": "prod",
    }
    cfg = TradingConfig.from_env(env)
    assert cfg.paper is False

