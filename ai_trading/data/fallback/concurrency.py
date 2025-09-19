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


_ASYNCIO_PRIMITIVE_NAMES = {"Lock", "Semaphore", "Event", "Condition", "BoundedSemaphore"}
_ASYNCIO_LOCK_NAMES = {"Lock", "Semaphore", "BoundedSemaphore"}


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


def _maybe_recreate_lock(obj: object, loop: asyncio.AbstractEventLoop) -> object:
    """Return a replacement lock/semaphore bound to ``loop`` when necessary."""

    obj_type = type(obj)
    type_name = obj_type.__name__

    if type_name not in _ASYNCIO_LOCK_NAMES:
        return obj

    bound_loop = None
    for attr_name in ("_loop", "_bound_loop"):
        if hasattr(obj, attr_name):
            bound_loop = getattr(obj, attr_name)
            if bound_loop is not None:
                break

    if bound_loop is loop:
        return obj

    def _capture_sem_state() -> dict[str, int]:
        state: dict[str, int] = {}
        for attr_name in ("_value", "_initial_value", "_bound_value"):
            value = getattr(obj, attr_name, None)
            if isinstance(value, int):
                state[attr_name] = value
        return state

    if isinstance(obj, asyncio.Lock):
        return asyncio.Lock()

    bounded_semaphore_type = getattr(asyncio, "BoundedSemaphore", None)
    if bounded_semaphore_type is not None and isinstance(obj, bounded_semaphore_type):
        sem_state = _capture_sem_state()
        candidate_value = None
        for attr_name in ("_initial_value", "_bound_value", "_value"):
            value = sem_state.get(attr_name)
            if isinstance(value, int) and value >= 0:
                candidate_value = value
                break
        if candidate_value is None:
            candidate_value = 1
        replacement = bounded_semaphore_type(candidate_value)
        for attr_name, value in sem_state.items():
            if hasattr(replacement, attr_name) and isinstance(value, int) and value >= 0:
                setattr(replacement, attr_name, value)
        return replacement

    if isinstance(obj, asyncio.Semaphore):
        sem_state = _capture_sem_state()
        candidate_value = None
        for attr_name in ("_initial_value", "_value"):
            value = sem_state.get(attr_name)
            if isinstance(value, int) and value >= 0:
                candidate_value = value
                break
        if candidate_value is None:
            candidate_value = 1
        replacement = asyncio.Semaphore(candidate_value)
        for attr_name, value in sem_state.items():
            if hasattr(replacement, attr_name) and isinstance(value, int) and value >= 0:
                setattr(replacement, attr_name, value)
        return replacement

    constructor = getattr(asyncio, type_name, None)
    if callable(constructor):
        try:
            return constructor()
        except TypeError:
            pass
    try:
        return obj_type()
    except TypeError:
        return obj

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

    def _scan(obj, seen: set[int]) -> object:
        obj_id = id(obj)
        if obj_id in seen:
            return obj
        seen.add(obj_id)
        if _is_asyncio_primitive(obj):
            replacement = _maybe_recreate_lock(obj, loop)
            return replacement
        if isinstance(obj, Mapping):
            supports_setitem = hasattr(obj, "__setitem__")
            items = list(obj.items())
            mutated = False
            replacements: list[tuple[object, object]] = []
            for key, value in items:
                new_value = _scan(value, seen)
                replacements.append((key, new_value))
                if new_value is not value:
                    mutated = True
                    if supports_setitem:
                        try:
                            obj[key] = new_value
                        except Exception:
                            pass
            if mutated and not supports_setitem:
                mapping_type = type(obj)
                try:
                    return mapping_type(replacements)
                except Exception:
                    return dict(replacements)
            return obj
        if isinstance(obj, (list, tuple, set, frozenset)):
            if isinstance(obj, list):
                for idx, value in enumerate(list(obj)):
                    new_value = _scan(value, seen)
                    if new_value is not value:
                        obj[idx] = new_value
                return obj
            if isinstance(obj, tuple):
                mutated = False
                new_items = []
                for value in obj:
                    new_value = _scan(value, seen)
                    mutated = mutated or new_value is not value
                    new_items.append(new_value)
                if mutated:
                    return tuple(new_items)
                return obj
            if isinstance(obj, set):
                mutated = False
                new_values = []
                for value in list(obj):
                    new_value = _scan(value, seen)
                    mutated = mutated or new_value is not value
                    new_values.append(new_value)
                if mutated:
                    obj.clear()
                    obj.update(new_values)
                return obj
            mutated = False
            new_values = []
            for value in obj:
                new_value = _scan(value, seen)
                mutated = mutated or new_value is not value
                new_values.append(new_value)
            if mutated:
                return type(obj)(new_values)
            return obj
        if is_dataclass(obj) and not isinstance(obj, type):
            for field in fields(obj):
                try:
                    value = getattr(obj, field.name)
                except AttributeError:
                    continue
                new_value = _scan(value, seen)
                if new_value is not value:
                    try:
                        setattr(obj, field.name, new_value)
                    except Exception:
                        pass
            try:
                extra_attrs = vars(obj)
            except TypeError:
                extra_attrs = None
            if extra_attrs:
                for name, value in list(extra_attrs.items()):
                    new_value = _scan(value, seen)
                    if new_value is not value:
                        try:
                            setattr(obj, name, new_value)
                        except Exception:
                            pass
            return obj
        if isinstance(obj, SimpleNamespace):
            for name, value in list(vars(obj).items()):
                new_value = _scan(value, seen)
                if new_value is not value:
                    setattr(obj, name, new_value)
            return obj
        slots = getattr(type(obj), "__slots__", ())
        if slots:
            slot_names = (slots,) if isinstance(slots, str) else slots
            for name in slot_names:
                if hasattr(obj, name):
                    value = getattr(obj, name)
                    new_value = _scan(value, seen)
                    if new_value is not value:
                        try:
                            setattr(obj, name, new_value)
                        except Exception:
                            pass
            return obj
        module_name = getattr(obj.__class__, "__module__", "")
        if (
            hasattr(obj, "__dict__")
            and not isinstance(obj, ModuleType)
            and module_name.startswith(("ai_trading", "tests", "__main__"))
        ):
            for name, value in list(vars(obj).items()):
                new_value = _scan(value, seen)
                if new_value is not value:
                    try:
                        setattr(obj, name, new_value)
                    except Exception:
                        pass
            return obj

        return obj

    cells = getattr(worker, "__closure__", None)
    if cells:
        seen: set[int] = set()
        for cell in cells:
            try:
                original = cell.cell_contents
                new_value = _scan(original, seen)
                if new_value is not original:
                    cell.cell_contents = new_value
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
