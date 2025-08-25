from __future__ import annotations

import functools
import random
import time
from collections.abc import Callable
from typing import TypeVar, Literal

T = TypeVar("T")


def retry_call(
    func: Callable[..., T],
    *args,
    retries: int = 3,
    backoff: float = 0.25,
    max_backoff: float = 2.0,
    jitter: float = 0.1,
    exceptions: tuple[type[BaseException], ...] = (),
    sleep_fn=time.sleep,
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


def retry(
    retries: int = 3,
    delay: float = 0.5,
    backoff: float = 2.0,
    exceptions=(Exception,),
    *,
    mode: Literal["exponential", "fixed", "linear"] = "exponential",
    **kwargs,
):
    """Retry decorator supporting multiple backoff modes."""

    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kw):
            curr_delay = delay
            for attempt in range(retries):
                try:
                    return fn(*args, **kw)
                except exceptions:
                    if attempt == retries - 1:
                        raise
                    time.sleep(curr_delay)
                    if mode == "exponential":
                        curr_delay *= backoff
                    elif mode == "linear":
                        curr_delay += backoff
                    else:  # fixed
                        curr_delay = curr_delay
        return wrapper

    return decorator


__all__ = ["retry", "retry_call"]

