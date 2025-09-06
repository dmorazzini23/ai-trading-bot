from ai_trading.core.runtime import build_runtime
from ai_trading.config.management import TradingConfig
from ai_trading.position_sizing import get_max_position_size
from ai_trading.runtime.max_position_size import (
    get_max_position_size as runtime_get_max_position_size,
)


def test_runtime_uses_env_max_position_size(monkeypatch):
    monkeypatch.setenv("MAX_POSITION_SIZE", "2500")
    cfg = TradingConfig.from_env()
    runtime = build_runtime(cfg)
    assert runtime.params["MAX_POSITION_SIZE"] == 2500.0


def test_runtime_derives_max_position_size(monkeypatch):
    monkeypatch.delenv("MAX_POSITION_SIZE", raising=False)
    monkeypatch.delenv("AI_TRADING_MAX_POSITION_SIZE", raising=False)
    cfg = TradingConfig(capital_cap=0.04)
    runtime = build_runtime(cfg)
    assert runtime.params["MAX_POSITION_SIZE"] == 8000.0


def test_runtime_matches_position_sizing(monkeypatch):
    monkeypatch.delenv("MAX_POSITION_SIZE", raising=False)
    monkeypatch.delenv("AI_TRADING_MAX_POSITION_SIZE", raising=False)
    cfg = TradingConfig(capital_cap=0.04)
    runtime = build_runtime(cfg)
    assert runtime.params["MAX_POSITION_SIZE"] == get_max_position_size(cfg)


def test_runtime_get_max_position_size_prefers_config(monkeypatch):
    monkeypatch.setenv("MAX_POSITION_SIZE", "999")
    cfg = TradingConfig(max_position_size=1234)
    assert runtime_get_max_position_size(cfg) == 1234.0

