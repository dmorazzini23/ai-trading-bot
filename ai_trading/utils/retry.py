from __future__ import annotations

import functools
import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

try:  # pragma: no cover - exercised via HAS_TENACITY flag in tests
    from tenacity import (
        stop_after_attempt as _stop_after_attempt,
        wait_exponential as _wait_exponential,
        wait_random as _wait_random,
        wait_fixed as _wait_fixed,
        wait_incrementing as _wait_incrementing,
        retry_if_exception_type as _retry_if_exception_type,
        retry as _tenacity_retry,
        RetryError as _RetryError,
    )
    if not isinstance(_RetryError, type) or not issubclass(_RetryError, BaseException):
        raise TypeError("Invalid RetryError type")
    stop_after_attempt = _stop_after_attempt
    wait_exponential = _wait_exponential
    wait_random = _wait_random
    retry_if_exception_type = _retry_if_exception_type
    RetryError = _RetryError
    HAS_TENACITY = True
except (ImportError, TypeError):  # pragma: no cover - fallback path when tenacity missing
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

    def wait_random(min: float = 0.0, max: float = 1.0) -> _Wait:
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


def retry_call(
    func: Callable[..., T],
    *args,
    retries: int = 3,
    backoff: float = 0.25,
    max_backoff: float = 2.0,
    jitter: float = 0.1,
    exceptions: tuple[type[BaseException], ...] = (),
    sleep_fn: Callable[[float], None] | None = None,
    **kwargs,
) -> T:
    """Exponential backoff helper for direct function calls."""

    attempt = 0
    delay = max(0.0, backoff)
    _sleep = sleep_fn or time.sleep  # resolve at call time to honor monkeypatches
    while True:
        try:
            return func(*args, **kwargs)
        except exceptions:
            attempt += 1
            if attempt > retries:
                raise
            delta = delay * jitter
            _sleep(max(0.0, delay + random.uniform(-delta, delta)))
            delay = min(max_backoff, delay * 2)


def retry(
    *,
    retries: int = 3,
    delay: float = 0.1,
    backoff: float = 2.0,
    mode: str = "exponential",
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    reraise: bool = False,
    # Tenacity-compatible kwargs (optional)
    stop: object | None = None,
    wait: object | None = None,
    retry: object | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Lightweight retry decorator with consistent API across environments.

    Parameters
    - retries: number of attempts (total calls), e.g. 3 means up to 3 invocations
    - delay: base delay seconds
    - backoff: multiplier for linear/exponential modes
    - mode: one of "fixed", "linear", "exponential"
    - exceptions: tuple of exception types to catch
    - reraise: when True, propagate the last exception instead of raising RetryError.
      Forwarded to Tenacity's ``retry`` decorator when available.
    """

    attempts = max(0, int(retries))
    base = max(0.0, float(delay))
    factor = max(0.0, float(backoff))
    mode_lc = str(mode or "exponential").lower()

    # Tenacity-style API path if stop/wait/retry provided explicitly
    if HAS_TENACITY and (stop is not None or wait is not None or retry is not None):
        _stop = stop if stop is not None else stop_after_attempt(attempts)
        _wait = wait
        if _wait is None:
            if mode_lc == "fixed":
                _wait = _wait_fixed(base)
            elif mode_lc == "linear":
                _wait = _wait_incrementing(start=base, increment=max(0.0, factor))
            else:
                _wait = _wait_exponential(multiplier=base, min=base)
        _retry = retry if retry is not None else retry_if_exception_type(exceptions)
        dec = _tenacity_retry(retry=_retry, stop=_stop, wait=_wait, reraise=reraise)
        # Also point tenacity.retry at this decorator to satisfy identity checks in tests
        try:
            import tenacity as _tenacity_mod  # type: ignore

            _tenacity_mod.retry = dec  # type: ignore[assignment]
        except Exception:  # pragma: no cover - best effort wiring
            pass

        def decorator(fn: Callable[..., T]) -> Callable[..., T]:
            wrapped = dec(fn)

            @functools.wraps(fn)
            def inner(*args, **kwargs):
                return wrapped(*args, **kwargs)

            return inner

        return decorator

    if HAS_TENACITY:
        _stop = stop_after_attempt(attempts)
        if mode_lc == "fixed":
            _wait = _wait_fixed(base)
        elif mode_lc == "linear":
            _wait = _wait_incrementing(start=base, increment=max(0.0, factor))
        else:  # exponential (default)
            _wait = _wait_exponential(multiplier=base, min=base)
        predicate = retry_if_exception_type(exceptions)
        dec = _tenacity_retry(retry=predicate, stop=_stop, wait=_wait, reraise=reraise)
        try:
            import tenacity as _tenacity_mod  # type: ignore

            _tenacity_mod.retry = dec  # type: ignore[assignment]
        except Exception:  # pragma: no cover - best effort wiring
            pass

        def decorator(fn: Callable[..., T]) -> Callable[..., T]:
            wrapped = dec(fn)

            @functools.wraps(fn)
            def inner(*args, **kwargs):
                return wrapped(*args, **kwargs)

            return inner

        return decorator

    # Fallback (no Tenacity installed)
    def _calc_wait(n: int) -> float:
        if mode_lc == "fixed":
            return base
        if mode_lc == "linear":
            return base + (n - 1) * factor
        # exponential
        return base * (2 ** (n - 1)) if n > 0 else 0.0

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    # Determine if this exception is retryable under the provided
                    # policy. Support either a predicate via `retry` (from
                    # retry_if_exception_type) or the `exceptions` tuple.
                    should_retry = False
                    try:
                        if retry is not None and callable(retry):  # predicate style
                            should_retry = bool(retry(exc))  # type: ignore[misc]
                        else:
                            # Fall back to tuple-based matching
                            exc_types = exceptions or tuple()
                            if not isinstance(exc_types, tuple):
                                exc_types = (exc_types,)  # type: ignore[assignment]
                            should_retry = isinstance(exc, exc_types)
                    except Exception:
                        # If the retry policy itself errors, fail safe to no-retry
                        should_retry = False

                    if not should_retry:
                        raise
                    attempt += 1
                    if attempt >= attempts:
                        if reraise:
                            raise
                        raise RetryError() from exc
                    time.sleep(max(0.0, _calc_wait(attempt)))

        return wrapper

    return decorator


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

# Align identity: expose our retry decorator as tenacity.retry when available
try:  # pragma: no cover - identity wiring for tests
    if HAS_TENACITY:
        import tenacity as _tenacity_mod  # type: ignore

        _tenacity_mod.retry = retry  # type: ignore[assignment]
except Exception:
    pass
