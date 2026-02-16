"""Dependency-specific circuit breakers for broker/data operations."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Any, Deque

from ai_trading.core.errors import ErrorInfo


@dataclass(slots=True)
class BreakerState:
    failures: Deque[float] = field(default_factory=deque)
    opened_until: float = 0.0
    last_error_info: ErrorInfo | None = None
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))


class DependencyBreakers:
    """Track per-dependency failures and open circuits on repeated errors."""

    def __init__(self) -> None:
        self._states: dict[str, BreakerState] = {}
        self._lock = Lock()

    def _state(self, dependency: str) -> BreakerState:
        key = str(dependency)
        if key not in self._states:
            self._states[key] = BreakerState()
        return self._states[key]

    @staticmethod
    def _monotonic() -> float:
        import time

        return time.monotonic()

    def allow(self, dep: str) -> bool:
        now = self._monotonic()
        with self._lock:
            state = self._state(dep)
            return now >= state.opened_until

    def open_reason(self, dep: str) -> str | None:
        with self._lock:
            state = self._state(dep)
            if self._monotonic() < state.opened_until:
                return f"CIRCUIT_OPEN_{dep}"
        return None

    def record_success(self, dep: str) -> None:
        with self._lock:
            state = self._state(dep)
            state.failures.clear()
            state.opened_until = 0.0
            state.last_error_info = None
            state.last_updated = datetime.now(UTC)

    def record_failure(self, dep: str, error_info: ErrorInfo) -> None:
        now = self._monotonic()
        with self._lock:
            state = self._state(dep)
            state.failures.append(now)
            state.last_error_info = error_info
            state.last_updated = datetime.now(UTC)

            ten_min = 600.0
            while state.failures and now - state.failures[0] > ten_min:
                state.failures.popleft()

            fail_60s = sum(1 for ts in state.failures if now - ts <= 60.0)
            fail_10m = len(state.failures)
            if fail_10m >= 10:
                state.opened_until = max(state.opened_until, now + 900.0)
            elif fail_60s >= 3:
                state.opened_until = max(state.opened_until, now + 60.0)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        now = self._monotonic()
        with self._lock:
            out: dict[str, dict[str, Any]] = {}
            for dep, state in self._states.items():
                out[dep] = {
                    "open": now < state.opened_until,
                    "opened_until": state.opened_until,
                    "failure_count_window": len(state.failures),
                    "last_reason_code": (
                        state.last_error_info.reason_code if state.last_error_info else None
                    ),
                }
            return out
