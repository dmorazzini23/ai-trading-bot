from __future__ import annotations

import threading

_lock = threading.RLock()


def optimize(params):
    """Return a shallow copy of params within a lock."""  # AI-AGENT-REF
    with _lock:
        return dict(params or {})


class AlgorithmOptimizer:
    """Minimal thread-safe optimizer stub."""  # AI-AGENT-REF

    def __init__(self):
        self._lock = threading.RLock()

    def _calculate_kelly_fraction(self, symbol: str) -> float:
        with self._lock:
            return 0.0


__all__ = ["optimize", "AlgorithmOptimizer"]

