from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from ai_trading.risk.engine import RiskEngine, TradeSignal
from ai_trading.strategy_allocator import StrategyAllocator, EPS

pytest.importorskip("numpy")


def test_available_exposure_respects_adaptive_cap() -> None:
    engine = RiskEngine()
    engine.exposure = {"equity": 0.2}
    engine.strategy_exposure = {"momentum": 0.1}
    engine.asset_limits["equity"] = 0.9
    engine.strategy_limits["momentum"] = 0.9
    engine._returns = [0.01, -0.005, 0.015, 0.0]
    signal = TradeSignal(
        symbol="AAPL",
        side="buy",
        confidence=0.7,
        strategy="momentum",
        weight=0.7,
        asset_class="equity",
    )
    available = engine._apply_weight_limits(signal)
    assert available == pytest.approx(0.6, rel=1e-6)


def test_allocator_scaling_preserves_proportions(caplog: pytest.LogCaptureFixture) -> None:
    allocator = StrategyAllocator()
    sig1 = SimpleNamespace(weight=0.4, symbol="AAA")
    sig2 = SimpleNamespace(weight=0.4 + EPS / 2, symbol="BBB")
    unchanged = allocator._scale_buy_weights([sig1, sig2], 0.8)
    assert not unchanged
    assert sig1.weight == pytest.approx(0.4)
    assert sig2.weight == pytest.approx(0.4 + EPS / 2)

    heavy1 = SimpleNamespace(weight=0.6, symbol="CCC")
    heavy2 = SimpleNamespace(weight=0.3, symbol="DDD")
    caplog.set_level(logging.INFO)
    scaled = allocator._scale_buy_weights([heavy1, heavy2], 0.8)
    assert scaled
    assert heavy1.weight + heavy2.weight <= 0.8 + EPS
    ratio_before = 0.6 / 0.3
    ratio_after = heavy1.weight / heavy2.weight if heavy2.weight else 0.0
    assert pytest.approx(ratio_before, rel=1e-6) == ratio_after
    records = [rec for rec in caplog.records if rec.message.startswith("ALLOCATION_SCALED |")]
    assert records, "expected allocation scaling log"
