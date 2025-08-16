from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MockSignal:
    symbol: str
    action: str  # 'buy' or 'sell'
    confidence: float


class StrategyAllocator:
    """Minimal allocator with confidence normalization hook for tests."""  # AI-AGENT-REF: test strategy allocator

    @staticmethod
    def _normalize_confidence(c: float) -> float:
        """Clip confidence to [0,1]."""  # AI-AGENT-REF: normalization helper
        return max(0.0, min(1.0, c))

    def allocate(self, signals_by_strategy: dict[str, list[MockSignal]]):
        out: list[MockSignal] = []
        for sigs in (signals_by_strategy or {}).values():
            for s in sigs or []:
                out.append(
                    MockSignal(
                        s.symbol, s.action, self._normalize_confidence(s.confidence)
                    )
                )
        return out


__all__ = ["MockSignal", "StrategyAllocator"]
