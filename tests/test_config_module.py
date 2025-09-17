import importlib
import os

import pytest


def _reload_config(monkeypatch, **env):
    module_name = "ai_trading.config"
    if module_name in list(importlib.sys.modules):
        del importlib.sys.modules[module_name]
    for k in ["MAX_DRAWDOWN_THRESHOLD", "TRADING_MODE", "KELLY_FRACTION", "CONF_THRESHOLD", "MAX_POSITION_SIZE"]:
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
    with pytest.raises(RuntimeError):
        _reload_config(monkeypatch)


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
