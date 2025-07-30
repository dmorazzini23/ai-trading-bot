"""
Strategy allocation with proper signal confirmation and hold protection.
"""
import logging
from typing import Dict, List, Any

try:
    from config import CONFIG
except ImportError:
    # Fallback config for testing environments
    from dataclasses import dataclass
    
    @dataclass
    class FallbackConfig:
        signal_confirmation_bars: int = 2
        delta_threshold: float = 0.02
    
    CONFIG = FallbackConfig()

logger = logging.getLogger(__name__)


class StrategyAllocator:
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.signal_history: Dict[str, List[float]] = {}
        self.last_direction: Dict[str, str] = {}
        self.last_confidence: Dict[str, float] = {}
        self.hold_protect: Dict[str, int] = {}

    def allocate(self, signals_by_strategy: Dict[str, List[Any]]) -> List[Any]:
        confirmed_signals = self._confirm_signals(signals_by_strategy)
        return self._allocate_confirmed(confirmed_signals)

    def _confirm_signals(self, signals_by_strategy: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        confirmed: Dict[str, List[Any]] = {}
        for strategy, signals in signals_by_strategy.items():
            confirmed[strategy] = []
            for s in signals:
                key = f"{s.symbol}_{s.side}"
                if key not in self.signal_history:
                    self.signal_history[key] = []

                self.signal_history[key].append(s.confidence)
                self.signal_history[key] = self.signal_history[key][-self.config.signal_confirmation_bars:]

                if len(self.signal_history[key]) >= self.config.signal_confirmation_bars:
                    avg_conf = sum(self.signal_history[key]) / len(self.signal_history[key])
                    if avg_conf > 0.6:
                        s.confidence = avg_conf
                        confirmed[strategy].append(s)
        return confirmed

    def _allocate_confirmed(self, confirmed_signals: Dict[str, List[Any]]) -> List[Any]:
        final_signals: List[Any] = []
        all_signals: List[Any] = []

        for strategy, signals in confirmed_signals.items():
            for s in signals:
                s.strategy = strategy
                all_signals.append(s)

        for s in sorted(all_signals, key=lambda x: (x.symbol, -x.confidence)):
            last_dir = self.last_direction.get(s.symbol)
            last_conf = self.last_confidence.get(s.symbol, 0.0)

            if last_dir and last_dir != s.side:
                if s.side == "sell" and self.hold_protect.get(s.symbol, 0) > 0:
                    remaining = self.hold_protect.get(s.symbol, 0)
                    logger.info(
                        "HOLD_PROTECT_ACTIVE",
                        extra={"symbol": s.symbol, "remaining_cycles": remaining},
                    )
                    self.hold_protect[s.symbol] = max(0, remaining - 1)
                    continue

            delta = abs(s.confidence - last_conf) if last_conf else s.confidence
            if delta < self.config.delta_threshold and last_dir == s.side:
                logger.info(
                    "SIGNAL_SKIPPED_DELTA",
                    extra={"symbol": s.symbol, "delta": delta, "threshold": self.config.delta_threshold},
                )
                continue

            self.last_direction[s.symbol] = s.side
            self.last_confidence[s.symbol] = s.confidence

            if s.side == "buy":
                self.hold_protect[s.symbol] = 4  # Changed from 3 to 4

            final_signals.append(s)

        return final_signals

    def update_reward(self, strategy: str, reward: float) -> None:
        """Update reward for a strategy (placeholder for test compatibility)."""
        logger.info(f"Strategy {strategy} reward updated: {reward}")


__all__ = ["StrategyAllocator"]

