import importlib
import os

import pytest


def _reload_config(monkeypatch, **env):
    module_name = "ai_trading.config"
    if module_name in list(importlib.sys.modules):
        del importlib.sys.modules[module_name]
    for k in [
        "MAX_DRAWDOWN_THRESHOLD",
        "AI_TRADING_MAX_DRAWDOWN_THRESHOLD",
        "TRADING_MODE",
        "KELLY_FRACTION",
        "CONF_THRESHOLD",
        "MAX_POSITION_SIZE",
    ]:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    return importlib.import_module(module_name)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    import sys
    sys.modules.pop("ai_trading.config", None)


def test_missing_drawdown_threshold(monkeypatch):
    module = _reload_config(monkeypatch)
    with pytest.raises(RuntimeError):
        module.get_max_drawdown_threshold()


def test_mode_presets(monkeypatch):
    cfg = _reload_config(monkeypatch, MAX_DRAWDOWN_THRESHOLD=0.2, TRADING_MODE="conservative")
    assert cfg.CONF_THRESHOLD == pytest.approx(0.85)
    assert cfg.MAX_POSITION_SIZE == pytest.approx(5000.0)

    cfg = _reload_config(monkeypatch, MAX_DRAWDOWN_THRESHOLD=0.2)
    assert cfg.CONF_THRESHOLD == pytest.approx(0.75)
    assert cfg.MAX_POSITION_SIZE == pytest.approx(8000.0)

    cfg = _reload_config(monkeypatch, MAX_DRAWDOWN_THRESHOLD=0.2, TRADING_MODE="aggressive")
    assert cfg.CONF_THRESHOLD == pytest.approx(0.65)
    assert cfg.MAX_POSITION_SIZE == pytest.approx(12000.0)


def test_balanced_mode_max_position_alias(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "balanced")
    monkeypatch.setenv("MAX_DRAWDOWN_THRESHOLD", "0.2")
    monkeypatch.delenv("MAX_POSITION_SIZE", raising=False)
    alias_value = "4321.5"
    monkeypatch.setenv("AI_TRADING_MAX_POSITION_SIZE", alias_value)

    from ai_trading.config.management import TradingConfig

    cfg = TradingConfig.from_env()
    assert cfg.max_position_size == pytest.approx(float(alias_value))


def test_trading_mode_overrides_restore_documented_defaults(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "aggressive")
    monkeypatch.setenv("MAX_DRAWDOWN_THRESHOLD", "0.2")
    for key in (
        "KELLY_FRACTION",
        "AI_TRADING_KELLY_FRACTION",
        "CONF_THRESHOLD",
        "AI_TRADING_CONF_THRESHOLD",
        "DAILY_LOSS_LIMIT",
        "AI_TRADING_DAILY_LOSS_LIMIT",
        "MAX_POSITION_SIZE",
        "AI_TRADING_MAX_POSITION_SIZE",
        "CAPITAL_CAP",
        "AI_TRADING_CAPITAL_CAP",
        "CONFIRMATION_COUNT",
        "TAKE_PROFIT_FACTOR",
        "AI_TRADING_TRADING_MODE",
    ):
        monkeypatch.delenv(key, raising=False)

    from ai_trading.config.management import TradingConfig

    expected = {
        "conservative": {
            "kelly_fraction": 0.25,
            "conf_threshold": 0.85,
            "daily_loss_limit": 0.03,
            "max_position_size": 5000.0,
            "capital_cap": 0.20,
            "confirmation_count": 3,
            "take_profit_factor": 1.5,
        },
        "balanced": {
            "kelly_fraction": 0.6,
            "conf_threshold": 0.75,
            "daily_loss_limit": 0.05,
            "max_position_size": 8000.0,
            "capital_cap": 0.25,
            "confirmation_count": 2,
            "take_profit_factor": 1.8,
        },
        "aggressive": {
            "kelly_fraction": 0.75,
            "conf_threshold": 0.65,
            "daily_loss_limit": 0.08,
            "max_position_size": 12000.0,
            "capital_cap": 0.30,
            "confirmation_count": 1,
            "take_profit_factor": 2.5,
        },
    }

    for mode, values in expected.items():
        cfg = TradingConfig.from_env(
            {
                "TRADING_MODE": mode,
                "MAX_DRAWDOWN_THRESHOLD": 0.2,
            }
        )
        assert cfg.kelly_fraction == pytest.approx(values["kelly_fraction"])
        assert cfg.conf_threshold == pytest.approx(values["conf_threshold"])
        assert cfg.daily_loss_limit == pytest.approx(values["daily_loss_limit"])
        assert cfg.max_position_size == pytest.approx(values["max_position_size"])
        assert cfg.capital_cap == pytest.approx(values["capital_cap"])
        assert cfg.confirmation_count == values["confirmation_count"]
        assert cfg.take_profit_factor == pytest.approx(values["take_profit_factor"])
