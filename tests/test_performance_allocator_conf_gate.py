import logging

import pytest
from ai_trading.config.management import TradingConfig
from ai_trading.config.settings import get_settings
from ai_trading.strategies.performance_allocator import (
    PerformanceBasedAllocator,
    _resolve_conf_threshold,
)


class Sig:
    def __init__(self, symbol: str, confidence: float):
        self.symbol = symbol
        self.confidence = confidence


def test_threshold_resolution_prefers_trading_config(monkeypatch):
    monkeypatch.setenv("SCORE_CONFIDENCE_MIN", "0.2")
    get_settings.cache_clear()  # AI-AGENT-REF: refresh settings after env change
    cfg = TradingConfig(score_confidence_min=0.75)
    assert _resolve_conf_threshold(cfg) == pytest.approx(0.75)


def test_allocator_confidence_gate_filters_and_logs(caplog):
    caplog.set_level(logging.INFO)
    alloc = PerformanceBasedAllocator()
    cfg = TradingConfig(score_confidence_min=0.7)
    inputs = {
        "momentum": [Sig("AAPL", 0.65), Sig("MSFT", 0.71), Sig("NVDA", 0.90)],
        "meanrev": [Sig("TSLA", 0.40), Sig("AMZN", 0.72)],
    }

    out = alloc.allocate(inputs, cfg)

    kept_symbols = {s.symbol for xs in out.values() for s in xs}
    assert kept_symbols == {"MSFT", "NVDA", "AMZN"}

    drops = [rec for rec in caplog.records if rec.message == "CONFIDENCE_DROP"]
    assert len(drops) >= 2
    for rec in drops:
        assert getattr(rec, "threshold", 0) == pytest.approx(0.7)

