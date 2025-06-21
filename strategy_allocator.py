import logging
import math
from typing import Dict, List

from strategies import TradeSignal
logger = logging.getLogger(__name__)
logger.debug("=== STRATEGY_ALLOCATOR LOADED === %s", __file__)


class StrategyAllocator:
    """Dynamic allocation of strategy weights."""

    def __init__(self) -> None:
        self.weights: Dict[str, float] = {}

    def update_reward(self, strategy: str, reward: float) -> None:
        # Validate inputs before updating internal state
        if not strategy:
            logger.warning("update_reward called with empty strategy name")
            return
        if not isinstance(reward, (int, float)) or math.isnan(reward):
            logger.warning("update_reward received invalid reward %r for %s", reward, strategy)
            return

        w = self.weights.get(strategy, 1.0)
        self.weights[strategy] = max(0.1, min(2.0, w + reward))
        logger.debug(
            "Updated weight for %s from %.3f to %.3f",
            strategy,
            w,
            self.weights[strategy],
        )

    def allocate(self, signals: Dict[str, List[TradeSignal]]) -> List[TradeSignal]:
        # Ensure signals dict is valid before processing
        if not isinstance(signals, dict) or not signals:
            logger.warning("allocate called with empty or invalid signals input")
            return []

        results: List[TradeSignal] = []
        for strat, sigs in signals.items():
            if not isinstance(sigs, list) or not sigs:
                logger.warning("No signals provided for strategy %s", strat)
                continue

            weight = self.weights.get(strat, 1.0)
            if not isinstance(weight, (int, float)) or math.isnan(weight) or weight <= 0:
                logger.warning(
                    "Invalid weight %r for strategy %s; defaulting to 1.0",
                    weight,
                    strat,
                )
                weight = 1.0

            before_count = len(results)
            for s in sigs:
                if not isinstance(s, TradeSignal):
                    logger.warning("Invalid TradeSignal %r in %s", s, strat)
                    continue
                if not isinstance(s.weight, (int, float)) or math.isnan(s.weight):
                    logger.warning("Signal weight invalid for %s; using default 1.0", s.symbol)
                    s.weight = 1.0
                # apply strategy weight
                s.weight *= weight
                results.append(s)

            logger.debug(
                "Allocated %d signals for %s with weight %.3f",
                len(results) - before_count,
                strat,
                weight,
            )

        return results


__all__ = ["StrategyAllocator"]
