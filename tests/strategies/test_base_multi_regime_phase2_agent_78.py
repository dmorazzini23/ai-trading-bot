from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai_trading.core.enums import RiskLevel
from ai_trading.order.types import OrderSide
from ai_trading.strategies.base import BaseStrategy, StrategyRegistry, StrategySignal


pd = pytest.importorskip("pandas")


class _ToyStrategy(BaseStrategy):
    def __init__(self) -> None:
        super().__init__("toy", "Toy", RiskLevel.CONSERVATIVE)
        self._signals: list[StrategySignal] = []

    def generate_signals(self, _market_data: dict) -> list[StrategySignal]:
        return list(self._signals)


def test_strategy_signal_validation_registry_and_context_generation() -> None:
    signal = StrategySignal("AAPL", OrderSide.BUY, strength=1.5, confidence=-1.0)
    assert signal.side == "buy"
    assert signal.strength == 1.0
    assert signal.confidence == 0.0
    assert signal.score == 1.0

    strategy = _ToyStrategy()
    assert strategy.validate_signal(StrategySignal("AAPL", "buy", 0.5, 0.7)) is True
    assert strategy.validate_signal(StrategySignal("", "buy", 0.5, 0.7)) is False
    assert strategy.validate_signal(StrategySignal("AAPL", "buy")) is False
    assert strategy.generate(SimpleNamespace(market_data={"x": 1})) == []

    strategy.update_performance(0.10, True)
    strategy.update_performance(-0.05, False)
    assert strategy.get_performance_summary()["win_rate"] == 0.5
    assert strategy.max_drawdown == -0.05

    registry = StrategyRegistry()
    strategy._signals = [
        StrategySignal("AAPL", "buy", 0.5, 0.7),
        StrategySignal("MSFT", "buy", 0.5, 0.1),
    ]
    assert registry.register_strategy(strategy) is True
    assert registry.register_strategy(strategy) is False
    assert registry.activate_strategy("toy") is True
    generated = registry.generate_signals_from_active_strategies({})
    assert [item.symbol for item in generated] == ["AAPL"]
    assert generated[0].strategy_id == "toy"
    assert strategy.signals_generated == 1
    assert registry.deactivate_strategy("toy") is True
    assert registry.unregister_strategy("toy") is True




