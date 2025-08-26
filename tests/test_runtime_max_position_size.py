from ai_trading.core.runtime import build_runtime
from ai_trading.config.management import TradingConfig


def test_runtime_uses_env_max_position_size(monkeypatch):
    monkeypatch.setenv("MAX_POSITION_SIZE", "2500")
    cfg = TradingConfig()
    runtime = build_runtime(cfg)
    assert runtime.params["MAX_POSITION_SIZE"] == 2500.0


def test_runtime_derives_max_position_size(monkeypatch):
    monkeypatch.delenv("MAX_POSITION_SIZE", raising=False)
    monkeypatch.delenv("AI_TRADING_MAX_POSITION_SIZE", raising=False)
    cfg = TradingConfig(capital_cap=0.04)
    runtime = build_runtime(cfg)
    assert runtime.params["MAX_POSITION_SIZE"] == 8000.0

