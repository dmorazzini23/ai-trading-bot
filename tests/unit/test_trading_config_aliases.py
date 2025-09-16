import logging

from ai_trading.config.management import TradingConfig
from ai_trading.main import _fail_fast_env


def test_trading_config_env_aliases():
    env = {
        "DAILY_LOSS_LIMIT": "0.06",
        "CAPITAL_CAP": "0.05",
        "MAX_POSITION_MODE": "STATIC",
        "DATA_FEED": "iex",
        "DATA_PROVIDER": "alpaca",
        "PAPER": "true",
        "MAX_DRAWDOWN_THRESHOLD": "0.15",
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
    env = {
        "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
        "MAX_DRAWDOWN_THRESHOLD": "0.15",
    }
    cfg = TradingConfig.from_env(env)
    assert cfg.paper is True


def test_paper_false_when_live_prod():
    env = {
        "ALPACA_BASE_URL": "https://api.alpaca.markets",
        "APP_ENV": "prod",
        "MAX_DRAWDOWN_THRESHOLD": "0.15",
    }
    cfg = TradingConfig.from_env(env)
    assert cfg.paper is False


def test_fail_fast_env_warns_on_alias_override(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="ai_trading.main")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_DATA_FEED", "iex")
    monkeypatch.setenv("WEBHOOK_SECRET", "hook")
    monkeypatch.setenv("CAPITAL_CAP", "0.25")
    monkeypatch.setenv("ALPACA_API_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("DOLLAR_RISK_LIMIT", "0.10")
    monkeypatch.setenv("DAILY_LOSS_LIMIT", "0.02")

    _fail_fast_env()

    records = [
        record
        for record in caplog.records
        if record.name == "ai_trading.main" and record.message == "DOLLAR_RISK_LIMIT_ALIAS_OVERRIDE"
    ]
    assert records, "Expected DOLLAR_RISK_LIMIT_ALIAS_OVERRIDE warning"
    record = records[0]
    assert getattr(record, "env_value", None) == "0.10"
    assert getattr(record, "trading_config_value", None) == 0.02

