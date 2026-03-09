from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from importlib import metadata
import types
from typing import Any, TypeVar, cast

T = TypeVar("T")


def _is_real_tenacity(mod: types.ModuleType) -> bool:
    """Return True only when the imported module is the real PyPI package.

    Some tests install lightweight stubs into ``sys.modules["tenacity"]`` to
    keep optional imports working under minimal environments. Those stubs may
    export a ``retry`` decorator that is a no-op, which breaks retry semantics.
    """

    try:
        metadata.version("tenacity")
    except metadata.PackageNotFoundError:
        return False
    except Exception:
        return False

    try:
        path = inspect.getfile(mod)
    except Exception:
        return False

    path_lc = path.lower()
    return ("site-packages" in path_lc) or ("dist-packages" in path_lc) or ("/tmp" in path_lc)


_tenacity_retry: Any = None
_stop_after_attempt: Callable[[int], object]
_wait_fixed: Callable[[float], object]
_retry_if_exception_type: Callable[[tuple[type[BaseException], ...]], object]


def _fallback_stop_after_attempt(max_attempts: int) -> Callable[[int], bool]:
    def stop(attempt: int) -> bool:
        return attempt >= max_attempts

    return stop


def _fallback_wait_fixed(delay: float) -> Callable[[int], float]:
    def wait(_attempt: int) -> float:
        return delay

    return wait


def _fallback_retry_if_exception_type(
    exc_types: tuple[type[BaseException], ...]
) -> Callable[[BaseException], bool]:
    def predicate(exc: BaseException) -> bool:
        return isinstance(exc, exc_types)

    return predicate


try:  # pragma: no cover - optional tenacity import
    import tenacity as _tenacity_mod

    from tenacity import (
        retry as _tenacity_retry_impl,
        stop_after_attempt as _stop_after_attempt_impl,
        wait_fixed as _wait_fixed_impl,
        retry_if_exception_type as _retry_if_exception_type_impl,
    )
    if not _is_real_tenacity(_tenacity_mod):
        raise ImportError("tenacity stub detected")
    _tenacity_retry = _tenacity_retry_impl
    _stop_after_attempt = _stop_after_attempt_impl
    _wait_fixed = _wait_fixed_impl
    _retry_if_exception_type = _retry_if_exception_type_impl
    HAS_TENACITY = True
except Exception:  # pragma: no cover - tenacity missing
    HAS_TENACITY = False
    _stop_after_attempt = _fallback_stop_after_attempt
    _wait_fixed = _fallback_wait_fixed
    _retry_if_exception_type = _fallback_retry_if_exception_type


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
        # ``retry_error_cls=None`` can surface the last exception in some
        # tenacity versions. Keep behavior stable by always returning fallback
        # when retry budget is exhausted.
        tenacity_attempts = max(1, attempts)
        dec = _tenacity_retry(
            stop=_stop_after_attempt(tenacity_attempts),
            wait=_wait_fixed(base_delay),
            retry=_retry_if_exception_type(exceptions),
            reraise=False,
            retry_error_callback=lambda _state: fallback,
        )

        def tenacity_decorator(fn: Callable[..., T]) -> Callable[..., T]:
            wrapped = dec(fn)
            return cast(Callable[..., T], wrapped)

        return tenacity_decorator

    def fallback_decorator(fn: Callable[..., T]) -> Callable[..., T]:
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

    return fallback_decorator


__all__ = ["retry_mode", "HAS_TENACITY"]
