import types

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
