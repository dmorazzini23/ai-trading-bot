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
    timeout: float | None = None,
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
    timeout:
        Optional per-task timeout in seconds. ``None`` disables the timeout.

    Returns
    -------
    tuple[dict[str, T | None], set[str], set[str]]
        A tuple of ``(results, successful, failed)`` where ``results`` maps
        symbols to worker return values or ``None``.
    """

    SUCCESSFUL_SYMBOLS.clear()
    FAILED_SYMBOLS.clear()

    results: dict[str, T | None] = {}

    loop = asyncio.get_running_loop()

    def _scan(obj, seen: set[int]) -> None:
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)
        if isinstance(obj, (asyncio.Lock, asyncio.Semaphore, asyncio.Event, asyncio.Condition)):
            try:
                obj._loop = loop  # type: ignore[attr-defined]
            except Exception:
                pass
            return
        if isinstance(obj, dict):
            for value in obj.values():
                _scan(value, seen)
            return
        if isinstance(obj, (list, tuple, set)):
            for value in obj:
                _scan(value, seen)
            return

    cells = getattr(worker, "__closure__", None)
    if cells:
        seen: set[int] = set()
        for cell in cells:
            try:
                _scan(cell.cell_contents, seen)
            except ValueError:
                continue

    sem = asyncio.Semaphore(max(1, max_concurrency))

    async def _run(sym: str) -> None:
        res: T | None = None
        try:
            async with sem:
                coro = worker(sym)
                if timeout is not None:
                    coro = asyncio.wait_for(coro, timeout)
                res = await coro
        except asyncio.CancelledError:  # pragma: no cover - cancel treated as failure
            res = None
        except Exception:  # pragma: no cover - worker errors become None
            res = None
        finally:
            results[sym] = res
            if res is None:
                FAILED_SYMBOLS.add(sym)
            else:
                SUCCESSFUL_SYMBOLS.add(sym)

    tasks = [asyncio.create_task(_run(sym)) for sym in symbols]
    await asyncio.gather(*tasks, return_exceptions=True)
    return results, SUCCESSFUL_SYMBOLS.copy(), FAILED_SYMBOLS.copy()


__all__ = ["run_with_concurrency", "SUCCESSFUL_SYMBOLS", "FAILED_SYMBOLS"]
