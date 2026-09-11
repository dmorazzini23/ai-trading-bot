"""Retry helpers for idempotent dependency reads."""
from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS
from alpaca.common.exceptions import APIError

import random
import time
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from ai_trading.core.dependency_breakers import DependencyBreakers
from ai_trading.core.errors import ErrorAction, ErrorInfo

T = TypeVar("T")
BROKER_READ_EXCEPTIONS: tuple[type[Exception], ...] = (*AI_TRADING_FALLBACK_EXCEPTIONS, APIError)


def retry_idempotent(
    fn: Callable[[], T],
    *,
    dep: str,
    breakers: DependencyBreakers,
    classify_exception: Callable[..., ErrorInfo],
    max_attempts: int = 3,
    max_total_seconds: float = 5.0,
    base_delay: float = 0.2,
    jitter: float = 0.1,
    context: Mapping[str, Any] | None = None,
) -> T:
    """Retry idempotent operations with dependency-aware breaker updates."""

    attempts = 0
    start = time.monotonic()
    symbol = str((context or {}).get("symbol") or "") or None
    sleeve = str((context or {}).get("sleeve") or "") or None

    while True:
        try:
            result = fn()
            breakers.record_success(dep)
            return result
        except BROKER_READ_EXCEPTIONS as exc:
            attempts += 1
            info = classify_exception(
                exc,
                dependency=dep,
                symbol=symbol,
                sleeve=sleeve,
            )
            breakers.record_failure(dep, info)

            elapsed = time.monotonic() - start
            exhausted = attempts >= max(1, int(max_attempts)) or elapsed >= max_total_seconds
            if exhausted or not info.retryable or info.action is not ErrorAction.RETRY:
                raise

            delay = max(0.0, float(base_delay)) * (2 ** max(0, attempts - 1))
            if jitter > 0:
                delay += random.uniform(0.0, float(jitter))
            remaining = max_total_seconds - elapsed
            if remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
