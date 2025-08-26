from __future__ import annotations

import functools
import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

try:  # pragma: no cover - exercised via HAS_TENACITY flag in tests
    from tenacity import (
        retry as _retry,
        stop_after_attempt as _stop_after_attempt,
        wait_exponential as _wait_exponential,
        wait_random as _wait_random,
        retry_if_exception_type as _retry_if_exception_type,
        RetryError as _RetryError,
    )
    if not isinstance(_RetryError, type) or not issubclass(_RetryError, BaseException):
        raise TypeError("Invalid RetryError type")
    retry = _retry
    stop_after_attempt = _stop_after_attempt
    wait_exponential = _wait_exponential
    wait_random = _wait_random
    retry_if_exception_type = _retry_if_exception_type
    RetryError = _RetryError
    HAS_TENACITY = True
except Exception:  # pragma: no cover - fallback path when tenacity missing
    HAS_TENACITY = False

    class RetryError(Exception):
        """Fallback RetryError used when Tenacity is unavailable."""

    def stop_after_attempt(max_attempts: int) -> Callable[[int], bool]:
        def stop(attempt_number: int) -> bool:
            return attempt_number >= max_attempts
        return stop

    class _Wait:
        def __init__(self, fn: Callable[[int], float]):
            self._fn = fn

        def __call__(self, attempt: int) -> float:
            return self._fn(attempt)

        def __add__(self, other: "_Wait") -> "_Wait":
            return _Wait(lambda a: self(a) + other(a))

    def wait_exponential(
        *,
        multiplier: float = 1.0,
        min: float = 0.0,
        max: float | None = None,
    ) -> _Wait:
        def fn(attempt: int) -> float:
            delay = multiplier * (2 ** (attempt - 1))
            delay = max(delay, min)
            if max is not None:
                delay = min(delay, max)
            return delay
        return _Wait(fn)

    def wait_random(*, min: float = 0.0, max: float = 1.0) -> _Wait:
        def fn(_attempt: int) -> float:
            return random.uniform(min, max)
        return _Wait(fn)

    def retry_if_exception_type(
        exc_types: type[BaseException] | tuple[type[BaseException], ...]
    ) -> Callable[[BaseException], bool]:
        if not isinstance(exc_types, tuple):
            exc_types = (exc_types,)

        def predicate(exc: BaseException) -> bool:
            return isinstance(exc, exc_types)

        return predicate

    def retry(
        *,
        retry: Callable[[BaseException], bool] | None = None,
        stop: Callable[[int], bool] | None = None,
        wait: Callable[[int], float] | None = None,
        reraise: bool = False,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(fn: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                attempt = 0
                while True:
                    try:
                        return fn(*args, **kwargs)
                    except Exception as exc:  # pragma: no cover - used in fallback
                        attempt += 1
                        if retry and not retry(exc):
                            raise
                        if stop and stop(attempt):
                            if reraise:
                                raise
                            raise RetryError() from exc
                        delay = wait(attempt) if wait else 0.0
                        time.sleep(delay)
            return wrapper
        return decorator


def retry_call(
    func: Callable[..., T],
    *args,
    retries: int = 3,
    backoff: float = 0.25,
    max_backoff: float = 2.0,
    jitter: float = 0.1,
    exceptions: tuple[type[BaseException], ...] = (),
    sleep_fn: Callable[[float], None] = time.sleep,
    **kwargs,
) -> T:
    """Exponential backoff helper for direct function calls."""

    attempt = 0
    delay = max(0.0, backoff)
    while True:
        try:
            return func(*args, **kwargs)
        except exceptions:
            attempt += 1
            if attempt > retries:
                raise
            delta = delay * jitter
            sleep_fn(max(0.0, delay + random.uniform(-delta, delta)))
            delay = min(max_backoff, delay * 2)


__all__ = [
    "retry",
    "stop_after_attempt",
    "wait_exponential",
    "wait_random",
    "retry_if_exception_type",
    "RetryError",
    "retry_call",
    "HAS_TENACITY",
]
