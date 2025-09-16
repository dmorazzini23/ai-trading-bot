import logging

import pytest

from ai_trading.config.management import TradingConfig
from ai_trading.main import _fail_fast_env


@pytest.mark.parametrize(
    ("env_updates", "expected_limit"),
    [
        pytest.param(
            {"DOLLAR_RISK_LIMIT": "0.06", "DAILY_LOSS_LIMIT": "0.02"},
            0.06,
            id="canonical_only",
        ),
        pytest.param(
            {"DOLLAR_RISK_LIMIT": "", "DAILY_LOSS_LIMIT": "0.07"},
            0.07,
            id="alias_only",
        ),
    ],
)
def test_trading_config_env_aliases(env_updates, expected_limit):
    env = {
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
    env.update(env_updates)

    cfg = TradingConfig.from_env(env)

    assert cfg.dollar_risk_limit == pytest.approx(expected_limit)
    assert cfg.capital_cap == 0.05
    assert cfg.max_position_mode == "STATIC"
    assert cfg.kelly_fraction_max == 0.2
    assert cfg.min_sample_size == 50
    assert cfg.confidence_level == 0.9
    if "DAILY_LOSS_LIMIT" in env_updates:
        assert cfg.daily_loss_limit == pytest.approx(
            float(env_updates["DAILY_LOSS_LIMIT"])
        )
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


def test_fail_fast_env_alias_override_logging(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="ai_trading.main")
    base_env = {
        "ALPACA_API_KEY": "key",
        "ALPACA_SECRET_KEY": "secret",
        "ALPACA_DATA_FEED": "iex",
        "WEBHOOK_SECRET": "hook",
        "CAPITAL_CAP": "0.25",
        "ALPACA_API_URL": "https://paper-api.alpaca.markets",
    }
    for key, value in base_env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("DAILY_LOSS_LIMIT", "0.02")

    def _alias_override_records():
        return [
            record
            for record in caplog.records
            if record.name == "ai_trading.main"
            and record.message == "DOLLAR_RISK_LIMIT_ALIAS_OVERRIDE"
        ]

    caplog.clear()
    monkeypatch.setenv("DOLLAR_RISK_LIMIT", "0.10")
    _fail_fast_env()
    assert not _alias_override_records()

    caplog.clear()
    monkeypatch.delenv("DOLLAR_RISK_LIMIT", raising=False)
    _fail_fast_env()

    records = _alias_override_records()
    assert records, "Expected DOLLAR_RISK_LIMIT_ALIAS_OVERRIDE warning when alias backfills"
    record = records[0]
    assert getattr(record, "env_value", None) == "0.02"
    assert getattr(record, "trading_config_value", None) == 0.02
