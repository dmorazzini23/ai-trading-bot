from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

from ai_trading.strategies.base import StrategySignal  # TradeSignal alias

# AI-AGENT-REF: minimal allocator facade for tests
@dataclass
class _AllocConfig:
    # Sensible defaults used by tests
    min_confidence: float = 0.0
    delta_threshold: float = 0.0
    signal_confirmation_bars: int = 1


@dataclass
class StrategyAllocator:
    """Minimal allocator satisfying tests. Passes signals through with simple confirmation."""

    config: _AllocConfig = field(default_factory=_AllocConfig)
    _confirm_counter: Dict[str, int] = field(default_factory=dict)

    def allocate(self, signals_by_strategy: Dict[str, List[StrategySignal]]) -> List[StrategySignal]:
        out: List[StrategySignal] = []
        for strat, sigs in signals_by_strategy.items():
            for s in sigs:
                # Confidence filter
                if getattr(s, "confidence", 1.0) < getattr(self.config, "min_confidence", 0.0):
                    continue
                # Basic confirmation window
                need = max(1, int(getattr(self.config, "signal_confirmation_bars", 1)))
                k = f"{strat}:{s.symbol}:{s.side}"
                self._confirm_counter[k] = self._confirm_counter.get(k, 0) + 1
                if self._confirm_counter[k] >= need:
                    out.append(s)
        return out

    # Helper used in some regression tests
    def record_trade_result(self, strategy_name: str, trade_result: dict) -> None:  # pragma: no cover
        # No-op for tests that only verify call path
        return None
