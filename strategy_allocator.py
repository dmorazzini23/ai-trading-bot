from typing import Dict
from typing import List
from strategies import TradeSignal


class StrategyAllocator:
    """Dynamic allocation of strategy weights."""

    def __init__(self) -> None:
        self.weights: Dict[str, float] = {}

    def update_reward(self, strategy: str, reward: float) -> None:
        w = self.weights.get(strategy, 1.0)
        self.weights[strategy] = max(0.1, min(2.0, w + reward))

    def allocate(self, signals: Dict[str, List[TradeSignal]]) -> List[TradeSignal]:
        results: List[TradeSignal] = []
        for strat, sigs in signals.items():
            weight = self.weights.get(strat, 1.0)
            for s in sigs:
                s.weight *= weight
                results.append(s)
        return results
