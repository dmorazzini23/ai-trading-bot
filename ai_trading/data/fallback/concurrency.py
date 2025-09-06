"""Async helpers for running fallback tasks with bounded concurrency.

The scheduler limits the number of in-flight tasks and aggregates their results
into per-symbol mappings and result sets.  It avoids spawning more than
``max_concurrency`` tasks at a time, preventing unbounded queue growth when
processing large symbol lists.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar

T = TypeVar("T")

# Result sets populated after ``run_with_concurrency`` completes.
SUCCESSFUL_SYMBOLS: set[str] = set()
FAILED_SYMBOLS: set[str] = set()


async def run_with_concurrency(
    symbols: Iterable[str],
    worker: Callable[[str], Awaitable[T]],
    *,
    max_concurrency: int = 4,
) -> tuple[dict[str, T | None], set[str], set[str]]:
    """Execute ``worker`` for each symbol with a concurrency limit.

    Parameters
    ----------
    symbols:
        Iterable of symbol strings to process.
    worker:
        Awaitable callable invoked with each symbol. Its return value becomes
        the entry in the result mapping. Exceptions are caught and return
        ``None`` entries.
    max_concurrency:
        Maximum number of tasks scheduled at any time.

    Returns
    -------
    tuple[dict[str, T | None], set[str], set[str]]
        A tuple of ``(results, successful, failed)`` where ``results`` maps
        symbols to worker return values or ``None``.
    """

    SUCCESSFUL_SYMBOLS.clear()
    FAILED_SYMBOLS.clear()

    results: dict[str, T | None] = {}
    iterator = iter(symbols)
    running: set[asyncio.Task[tuple[str, T | None]]] = set()

    async def _run(sym: str) -> tuple[str, T | None]:
        try:
            return sym, await worker(sym)
        except Exception:  # pragma: no cover - worker errors become None
            return sym, None

    def _schedule_next() -> bool:
        try:
            sym = next(iterator)
        except StopIteration:
            return False
        running.add(asyncio.create_task(_run(sym)))
        return True

    for _ in range(max_concurrency):
        if not _schedule_next():
            break

    while running:
        done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            running.remove(task)
            sym, res = task.result()
            results[sym] = res
            if res is None:
                FAILED_SYMBOLS.add(sym)
            else:
                SUCCESSFUL_SYMBOLS.add(sym)
            _schedule_next()

    return results, SUCCESSFUL_SYMBOLS.copy(), FAILED_SYMBOLS.copy()


__all__ = ["run_with_concurrency", "SUCCESSFUL_SYMBOLS", "FAILED_SYMBOLS"]
