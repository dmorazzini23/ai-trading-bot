from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MockSignal:
    symbol: str
    action: str  # 'buy' or 'sell'
    confidence: float


class StrategyAllocator:
    """Minimal allocator with confidence normalization hook for tests."""  # AI-AGENT-REF: test strategy allocator

    # AI-AGENT-REF: resolve threshold via config chain
    @staticmethod
    def _resolve_conf_threshold(cfg) -> float:
        for name in ("score_confidence_min", "min_confidence", "conf_threshold"):
            v = getattr(cfg, name, None)
            if isinstance(v, int | float) and 0 <= float(v) <= 1:
                return float(v)
        return 0.6

    @staticmethod
    def _normalize_confidence(x: float) -> float:
        try:
            return max(0.0, min(1.0, float(x)))
        except (TypeError, ValueError):
            return 0.0

    def __init__(self, config: Any | None = None):
        # AI-AGENT-REF: allow duck-typed config for tests
        self.config = config or type("_Cfg", (), {"score_confidence_min": 0.6})()

    def allocate(self, signals_by_strategy: dict[str, list[MockSignal]]):
        out: list[MockSignal] = []
        th = self._resolve_conf_threshold(self.config)
        for sigs in (signals_by_strategy or {}).values():
            for s in sigs or []:
                c = self._normalize_confidence(getattr(s, "confidence", 0.0))
                if c < th:
                    continue
                out.append(MockSignal(getattr(s, "symbol", ""), getattr(s, "action", "buy"), c))
        return out


__all__ = ["MockSignal", "StrategyAllocator"]
