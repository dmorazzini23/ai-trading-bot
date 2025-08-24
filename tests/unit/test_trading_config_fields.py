# AI-AGENT-REF: validate new TradingConfig fields and env overrides
from __future__ import annotations

from ai_trading.config.management import TradingConfig


def test_defaults_present():
    cfg = TradingConfig()
    assert cfg.kelly_fraction_max == 0.25
    assert cfg.min_sample_size == 10
    assert cfg.confidence_level == 0.90


def test_env_overrides_and_defaults(monkeypatch):
    monkeypatch.setenv('AI_TRADER_KELLY_FRACTION_MAX', '0.20')
    monkeypatch.setenv('AI_TRADER_MIN_SAMPLE_SIZE', '12')
    monkeypatch.setenv('AI_TRADER_CONFIDENCE_LEVEL', '0.85')
    cfg = TradingConfig.from_env()
    assert cfg.kelly_fraction_max == 0.20
    assert cfg.min_sample_size == 12
    assert cfg.confidence_level == 0.85
