import logging

import pytest

from ai_trading.strategies.performance_allocator import PerformanceBasedAllocator, _resolve_conf_threshold
from ai_trading.config.management import TradingConfig
from ai_trading.config.settings import get_settings
from ai_trading.settings import get_settings as base_get_settings


class Sig:
    def __init__(self, sym: str, confidence: float, weight: float = 1.0):
        self.symbol = sym
        self.confidence = confidence
        self.weight = weight


@pytest.fixture
def allocator():
    return PerformanceBasedAllocator()


def test_allocator_prefers_higher_conf_when_boost_enabled(monkeypatch, allocator, caplog):
    monkeypatch.setenv("SCORE_SIZE_MAX_BOOST", "1.15")
    monkeypatch.setenv("SCORE_SIZE_GAMMA", "1.0")
    get_settings.cache_clear()  # AI-AGENT-REF: refresh after env change
    base_get_settings.cache_clear()

    cfg = TradingConfig(score_confidence_min=0.70)
    th = _resolve_conf_threshold(cfg)

    caplog.set_level(logging.INFO)
    inputs = {"momentum": [Sig("A", th + 0.01), Sig("B", 0.95)]}

    out = allocator.allocate(inputs, cfg)

    items = list(out.get("momentum", []))
    assert len(items) == 2
    weights = {s.symbol: s.weight for s in items}
    assert weights["B"] > weights["A"]


