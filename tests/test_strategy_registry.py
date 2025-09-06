from ai_trading.strategies.base import BaseStrategy, StrategyRegistry
from ai_trading.core.enums import RiskLevel

class DummyStrategy(BaseStrategy):
    def generate_signals(self, market_data: dict) -> list:
        return []


def test_register_strategy_populates_position_size():
    registry = StrategyRegistry()
    strat = DummyStrategy("s1", "dummy", risk_level=RiskLevel.MODERATE)
    assert registry.register_strategy(strat)
    assert registry.strategy_performance[strat.strategy_id]["position_size"] > 0
