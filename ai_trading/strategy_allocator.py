from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
from ai_trading.strategies.base import StrategySignal as TradeSignal


@dataclass
class _AllocConfig:
    min_confidence: float = 0.0
    delta_threshold: float = 0.0
    signal_confirmation_bars: int = 1


@dataclass
class StrategyAllocator:
    config: _AllocConfig = field(default_factory=_AllocConfig)
    _confirm_counter: Dict[str, int] = field(default_factory=dict)

    def select_signals(self, signals_by_strategy: Dict[str, List[TradeSignal]]) -> List[TradeSignal]:
        out: List[TradeSignal] = []
        for strat, signals in (signals_by_strategy or {}).items():
            for s in signals or []:
                if getattr(s, "confidence", 1.0) < self.config.min_confidence:
                    continue
                need = max(1, int(self.config.signal_confirmation_bars))
                key = f"{strat}:{s.symbol}:{s.side}"
                self._confirm_counter[key] = self._confirm_counter.get(key, 0) + 1
                if self._confirm_counter[key] >= need:
                    out.append(s)
        return out
