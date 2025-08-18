"""Utility for retrying callables with bounded exponential backoff."""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def _nowait_sleep(seconds: float) -> None:
    """Sleep unless test flags indicate fast retry."""
    if os.getenv("PYTEST_RUNNING") == "1" or os.getenv("FAST_RETRY_IN_TESTS") == "1":
        return
    time.sleep(seconds)


def retry_call(
    func: Callable[..., T],
    *args: Any,
    exceptions: type[Exception] | tuple[type[Exception], ...],
    attempts: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 2.0,
    jitter: float = 0.1,
    **kwargs: Any,
) -> T:
    """Call ``func`` with retries on specified ``exceptions``.

    Applies exponential backoff with optional jitter between attempts. Sleep
    calls are skipped during testing when ``PYTEST_RUNNING`` or
    ``FAST_RETRY_IN_TESTS`` environment variables are set to ``"1"``.

    Args:
        func: Callable to execute.
        *args: Positional arguments for ``func``.
        exceptions: Exception type or tuple of types to trigger retry.
        attempts: Maximum number of attempts.
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay cap in seconds.
        jitter: Random jitter added to delay in seconds.
        **kwargs: Keyword arguments for ``func``.

    Returns:
        Result of ``func`` if successful.

    Raises:
        The last caught exception if all attempts fail.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    for attempt in range(1, attempts + 1):
        try:
            return func(*args, **kwargs)
        except exceptions as exc:
            if attempt >= attempts:
                raise exc
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            if jitter > 0:
                delay += random.uniform(0, jitter)
            _nowait_sleep(delay)
    # Should never reach here
    raise RuntimeError("Retry logic failed unexpectedly")
