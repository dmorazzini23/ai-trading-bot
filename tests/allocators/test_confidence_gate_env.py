import logging

from ai_trading.strategies.performance_allocator import (
    PerformanceBasedAllocator,
    AllocatorConfig,
)


def test_conf_gate_env_and_log(caplog):
    cfg = AllocatorConfig(score_confidence_min=0.70)
    alloc = PerformanceBasedAllocator(config=cfg)
    caplog.set_level(logging.INFO)
    w1 = alloc.score_to_weight(score=0.69)
    assert w1 == 0.0
    assert any(getattr(r, "threshold", None) == 0.7 for r in caplog.records)
    caplog.clear()
    alloc.score_to_weight(score=0.5)
    assert not any(getattr(r, "threshold", None) == 0.7 for r in caplog.records)
