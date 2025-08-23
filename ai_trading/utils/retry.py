from __future__ import annotations
import random
import time
from collections.abc import Callable
from typing import TypeVar
T = TypeVar('T')

def retry_call(func: Callable[..., T], *args, retries: int=3, backoff: float=0.25, max_backoff: float=2.0, jitter: float=0.1, exceptions: tuple[type[BaseException], ...]=(), sleep_fn=time.sleep, **kwargs) -> T:
    """
    Exponential backoff with jitter for transient exceptions.
    Retries on `exceptions`; re-raises otherwise.
    """
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