import ai_trading.config.alpaca as alpaca_cfg


def test_get_alpaca_config_strips_and_maps(monkeypatch):
    monkeypatch.setattr(alpaca_cfg, "ALPACA_AVAILABLE", False)

    class DummySettings:
        env = "dev"
        alpaca_base_url = None
        alpaca_rate_limit_per_min = None
        alpaca_data_feed = "iex"

    monkeypatch.setattr(alpaca_cfg, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(
        alpaca_cfg,
        "broker_keys",
        lambda: {" DEV_ALPACA_API_KEY ": " foo ", "DEV_ALPACA_SECRET_KEY": " bar "},
    )

    config = alpaca_cfg.get_alpaca_config()
    assert config.key_id == "foo"
    assert config.key == "foo"
    assert config.secret_key == "bar"


def test_get_alpaca_config_honors_execution_mode_live(monkeypatch):
    monkeypatch.setattr(alpaca_cfg, "ALPACA_AVAILABLE", False)
    monkeypatch.delenv("ALPACA_TRADING_BASE_URL", raising=False)
    monkeypatch.setenv("AI_TRADING_EXECUTION_MODE", "live")

    class DummySettings:
        env = "dev"
        execution_mode = "sim"
        alpaca_rate_limit_per_min = None
        alpaca_data_feed = "iex"

    monkeypatch.setattr(alpaca_cfg, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(
        alpaca_cfg,
        "broker_keys",
        lambda: {"ALPACA_API_KEY": "key", "ALPACA_SECRET_KEY": "secret"},
    )

    config = alpaca_cfg.get_alpaca_config()

    assert config.use_paper is False
    assert config.base_url == "https://api.alpaca.markets"


def test_get_alpaca_config_honors_explicit_live_base_url(monkeypatch):
    monkeypatch.setattr(alpaca_cfg, "ALPACA_AVAILABLE", False)
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://api.alpaca.markets")
    monkeypatch.setenv("EXECUTION_MODE", "paper")

    class DummySettings:
        env = "dev"
        execution_mode = "paper"
        alpaca_rate_limit_per_min = None
        alpaca_data_feed = "iex"

    monkeypatch.setattr(alpaca_cfg, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(
        alpaca_cfg,
        "broker_keys",
        lambda: {"ALPACA_API_KEY": "key", "ALPACA_SECRET_KEY": "secret"},
    )

    config = alpaca_cfg.get_alpaca_config()

    assert config.use_paper is False
    assert config.base_url == "https://api.alpaca.markets"


def test_get_alpaca_config_honors_settings_base_url(monkeypatch):
    monkeypatch.setattr(alpaca_cfg, "ALPACA_AVAILABLE", False)
    monkeypatch.delenv("ALPACA_TRADING_BASE_URL", raising=False)
    monkeypatch.setenv("EXECUTION_MODE", "paper")

    class DummySettings:
        env = "dev"
        execution_mode = "paper"
        alpaca_base_url = "https://api.alpaca.markets"
        alpaca_rate_limit_per_min = None
        alpaca_data_feed = "iex"

    monkeypatch.setattr(alpaca_cfg, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(
        alpaca_cfg,
        "broker_keys",
        lambda: {"ALPACA_API_KEY": "key", "ALPACA_SECRET_KEY": "secret"},
    )

    config = alpaca_cfg.get_alpaca_config()

    assert config.use_paper is False
    assert config.base_url == "https://api.alpaca.markets"
