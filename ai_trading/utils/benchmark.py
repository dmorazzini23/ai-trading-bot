from __future__ import annotations

"""Small benchmarking helpers.

This module provides lightweight timing utilities that rely on
``time.perf_counter_ns`` for high resolution measurements. Elapsed time is
reported in milliseconds.
"""

from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Any, Callable, TypeVar

T = TypeVar("T")


@dataclass
class BenchmarkTimer:
    """Context manager for measuring execution time.

    Example:
        >>> with BenchmarkTimer() as t:
        ...     heavy_operation()
        >>> t.elapsed_ms
        0.123  # milliseconds
    """

    start_ns: int = field(init=False, default=0)
    end_ns: int = field(init=False, default=0)
    elapsed_ms: float = field(init=False, default=0.0)

    def __enter__(self) -> "BenchmarkTimer":
        self.start_ns = perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.end_ns = perf_counter_ns()
        self.elapsed_ms = (self.end_ns - self.start_ns) / 1_000_000


def measure(func: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[T, float]:
    """Run ``func`` and return its result alongside elapsed milliseconds."""
    start = perf_counter_ns()
    result = func(*args, **kwargs)
    elapsed_ms = (perf_counter_ns() - start) / 1_000_000
    return result, elapsed_ms


__all__ = ["BenchmarkTimer", "measure"]
