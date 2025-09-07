from ai_trading.strategies.base import StrategyRegistry
from ai_trading.core.enums import RiskLevel
from tests.mocks.test_strategy import TestStrategy


def test_register_strategy_populates_position_size():
    registry = StrategyRegistry()
    strat = TestStrategy("s1", "dummy", risk_level=RiskLevel.MODERATE)
    assert registry.register_strategy(strat)
    assert registry.strategy_performance[strat.strategy_id]["position_size"] > 0
