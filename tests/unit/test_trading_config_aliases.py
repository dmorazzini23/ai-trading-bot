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
    assert str(snap["data"]["provider"]).startswith("alpaca")


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
    assert str(getattr(record, "env_value", None)) == "0.02"
    assert getattr(record, "trading_config_value", None) == 0.02


def test_max_position_size_canonical_value_wins_over_alias():
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
        "MAX_POSITION_SIZE": "7000",
        "AI_TRADING_MAX_POSITION_SIZE": "9000",
    }

    cfg = TradingConfig.from_env(env)

    assert cfg.max_position_size == pytest.approx(7000)


def test_explicit_mode_argument_overrides_trading_mode_env(monkeypatch):
    for key in (
        "TRADING_MODE",
        "bot_mode",
        "CAPITAL_CAP",
        "DAILY_LOSS_LIMIT",
        "DOLLAR_RISK_LIMIT",
        "KELLY_FRACTION",
        "CONF_THRESHOLD",
        "SIGNAL_CONFIRMATION_BARS",
        "TAKE_PROFIT_FACTOR",
        "MAX_POSITION_SIZE",
        "AI_TRADING_MAX_POSITION_SIZE",
    ):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("MAX_DRAWDOWN_THRESHOLD", "0.12")
    monkeypatch.setenv("TRADING_MODE", "aggressive")

    cfg = TradingConfig.from_env("conservative")

    assert cfg.capital_cap == pytest.approx(0.20)
    assert cfg.take_profit_factor == pytest.approx(1.5)
    assert cfg.max_position_size == pytest.approx(5000.0)
    assert cfg.conf_threshold == pytest.approx(0.85)


def test_fail_fast_env_health_port_conflict(monkeypatch):
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

    monkeypatch.setenv("RUN_HEALTHCHECK", "1")

    class DummySettings:
        api_port = 9001
        healthcheck_port = 9001

    monkeypatch.setattr("ai_trading.settings.get_settings", lambda: DummySettings())

    with pytest.raises(SystemExit) as excinfo:
        _fail_fast_env()

    message = str(excinfo.value)
    assert "HEALTHCHECK_PORT" in message
    assert "differ" in message
