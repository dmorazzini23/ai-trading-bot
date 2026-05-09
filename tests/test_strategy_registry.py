from ai_trading.strategies.base import StrategyRegistry
from ai_trading.core.enums import RiskLevel
from tests.mocks.test_strategy import TestStrategy


def test_register_strategy_populates_position_size():
    registry = StrategyRegistry()
    strat = TestStrategy("s1", "dummy", risk_level=RiskLevel.MODERATE)
    assert registry.register_strategy(strat)
    assert registry.strategy_performance[strat.strategy_id]["position_size"] > 0


def test_legacy_momentum_is_not_implicit_public_registry_default():
    from ai_trading.strategies import MomentumStrategy, REGISTRY

    assert REGISTRY.get("momentum") is None
    assert MomentumStrategy is not None
