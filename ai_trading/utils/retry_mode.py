from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

try:  # pragma: no cover - optional tenacity import
    from tenacity import (
        retry as _tenacity_retry,
        stop_after_attempt as _stop_after_attempt,
        wait_fixed as _wait_fixed,
        retry_if_exception_type as _retry_if_exception_type,
    )
    HAS_TENACITY = True
except Exception:  # pragma: no cover - tenacity missing
    HAS_TENACITY = False

    def _stop_after_attempt(max_attempts: int) -> Callable[[int], bool]:
        def stop(attempt: int) -> bool:
            return attempt >= max_attempts

        return stop

    def _wait_fixed(delay: float) -> Callable[[int], float]:
        def wait(_attempt: int) -> float:
            return delay

        return wait

    def _retry_if_exception_type(exc_types: tuple[type[BaseException], ...]) -> Callable[[BaseException], bool]:
        def predicate(exc: BaseException) -> bool:
            return isinstance(exc, exc_types)

        return predicate


def retry_mode(
    *,
    retries: int = 3,
    delay: float = 0.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    fallback: T | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry wrapper that returns ``fallback`` after exhausting attempts.

    When Tenacity is available, it is configured to stop after ``retries`` and
    to return ``fallback`` instead of raising :class:`tenacity.RetryError`.
    A minimal fallback implementation is provided when Tenacity isn't present.
    """

    attempts = max(0, int(retries))
    base_delay = max(0.0, float(delay))

    if HAS_TENACITY:
        dec = _tenacity_retry(
            stop=_stop_after_attempt(attempts),
            wait=_wait_fixed(base_delay),
            retry=_retry_if_exception_type(exceptions),
            retry_error_cls=None,
            retry_error_callback=lambda _state: fallback,
        )

        def decorator(fn: Callable[..., T]) -> Callable[..., T]:
            wrapped = dec(fn)
            return wrapped

        return decorator

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except exceptions:
                    attempt += 1
                    if attempt >= attempts:
                        return fallback
                    time.sleep(base_delay)

        return wrapper

    return decorator


__all__ = ["retry_mode", "HAS_TENACITY"]
