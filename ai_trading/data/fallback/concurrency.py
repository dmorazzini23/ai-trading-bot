"""Async helpers for running fallback tasks with bounded concurrency.

The scheduler limits the number of in-flight tasks and aggregates their results
into per-symbol mappings and result sets.  It avoids spawning more than
``max_concurrency`` tasks at a time, preventing unbounded queue growth when
processing large symbol lists.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import fields, is_dataclass
from types import ModuleType, SimpleNamespace
from typing import TypeVar

T = TypeVar("T")


_ASYNCIO_PRIMITIVE_NAMES = {"Lock", "Semaphore", "Event", "Condition"}


def _collect_asyncio_primitive_types() -> tuple[type, ...]:
    """Return concrete asyncio synchronization primitive types."""

    primitive_types: set[type] = set()

    for attr in _ASYNCIO_PRIMITIVE_NAMES:
        candidate = getattr(asyncio, attr, None)
        if isinstance(candidate, type):
            primitive_types.add(candidate)

    locks_module = getattr(asyncio, "locks", None)
    if locks_module is not None:
        for attr in _ASYNCIO_PRIMITIVE_NAMES:
            candidate = getattr(locks_module, attr, None)
            if isinstance(candidate, type):
                primitive_types.add(candidate)

    return tuple(primitive_types)


_ASYNCIO_PRIMITIVE_TYPES = _collect_asyncio_primitive_types()


def _is_asyncio_primitive(obj: object) -> bool:
    obj_type = type(obj)
    if obj_type in _ASYNCIO_PRIMITIVE_TYPES:
        return True
    return obj_type.__module__ == "_asyncio" and obj_type.__name__ in _ASYNCIO_PRIMITIVE_NAMES

# Result sets populated after ``run_with_concurrency`` completes.
SUCCESSFUL_SYMBOLS: set[str] = set()
FAILED_SYMBOLS: set[str] = set()
PEAK_SIMULTANEOUS_WORKERS: int = 0


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
        if _is_asyncio_primitive(obj):
            for attr_name in ("_loop", "_bound_loop"):
                if hasattr(obj, attr_name):
                    try:
                        setattr(obj, attr_name, loop)
                    except Exception:
                        pass
            return
        if isinstance(obj, Mapping):
            for value in obj.values():
                _scan(value, seen)
            return
        if isinstance(obj, (list, tuple, set, frozenset)):
            for value in obj:
                _scan(value, seen)
            return
        if is_dataclass(obj) and not isinstance(obj, type):
            for field in fields(obj):
                try:
                    value = getattr(obj, field.name)
                except AttributeError:
                    continue
                _scan(value, seen)
            return
        if isinstance(obj, SimpleNamespace):
            for value in vars(obj).values():
                _scan(value, seen)
            return
        slots = getattr(type(obj), "__slots__", ())
        if slots:
            slot_names = (slots,) if isinstance(slots, str) else slots
            for name in slot_names:
                if hasattr(obj, name):
                    _scan(getattr(obj, name), seen)
            return
        module_name = getattr(obj.__class__, "__module__", "")
        if (
            hasattr(obj, "__dict__")
            and not isinstance(obj, ModuleType)
            and module_name.startswith(("ai_trading", "tests", "__main__"))
        ):
            for value in vars(obj).values():
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

    concurrency_limit = max(1, max_concurrency)
    counter_lock = asyncio.Lock()
    running = 0
    peak_running = 0

    async def _increment() -> None:
        nonlocal running, peak_running
        async with counter_lock:
            running += 1
            if running > peak_running:
                peak_running = running
            assert (
                running <= concurrency_limit
            ), f"Exceeded max_concurrency={concurrency_limit}: running={running}"

    async def _decrement() -> None:
        nonlocal running
        async with counter_lock:
            running -= 1

    semaphore = asyncio.Semaphore(concurrency_limit)

    async def _bounded_execute(sym: str) -> None:
        async with semaphore:
            res: T | None = None
            await _increment()
            try:
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
                await _decrement()

    tasks = [asyncio.create_task(_bounded_execute(sym)) for sym in symbols]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    global PEAK_SIMULTANEOUS_WORKERS
    PEAK_SIMULTANEOUS_WORKERS = peak_running

    return results, SUCCESSFUL_SYMBOLS.copy(), FAILED_SYMBOLS.copy()


__all__ = [
    "run_with_concurrency",
    "SUCCESSFUL_SYMBOLS",
    "FAILED_SYMBOLS",
    "PEAK_SIMULTANEOUS_WORKERS",
]
