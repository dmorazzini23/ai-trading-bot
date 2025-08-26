# AI-AGENT-REF: validate new TradingConfig fields and env overrides
from __future__ import annotations

import pytest
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


def test_update_and_to_dict():
    cfg = TradingConfig()
    cfg.update(kelly_fraction_max=0.5, min_sample_size=20)
    assert cfg.kelly_fraction_max == 0.5
    assert cfg.min_sample_size == 20
    snap = cfg.to_dict()
    assert snap["kelly_fraction_max"] == 0.5
    assert snap["min_sample_size"] == 20
    assert snap["seed"] == cfg.seed


def test_update_unknown_key():
    cfg = TradingConfig()
    with pytest.raises(AttributeError):
        cfg.update(nonexistent=1)  # type: ignore[arg-type]
