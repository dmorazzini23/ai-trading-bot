"""Async helpers for running fallback tasks with bounded concurrency.

The closure-rebinding logic below walks dataclasses, ``SimpleNamespace``
instances, standard containers, and slot-based objects to recreate any
foreign-loop asyncio locks on the currently running loop. If additional
container types start capturing asyncio primitives, keep ``_scan`` in sync
with those structures.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping, MutableMapping
from dataclasses import fields, is_dataclass, replace
from types import MappingProxyType, ModuleType, SimpleNamespace
from typing import TypeVar

try:
    from ai_trading.http.pooling import get_host_limit as _pooling_host_limit
except Exception:  # pragma: no cover - pooling optional during stubbed tests
    def _get_effective_host_limit() -> int | None:
        return None
else:
    def _get_effective_host_limit() -> int | None:
        try:
            limit = int(_pooling_host_limit())
        except Exception:
            return None
        return max(1, limit)

T = TypeVar("T")

_ASYNCIO_PRIMITIVE_NAMES = {"Lock", "Semaphore", "Event", "Condition", "BoundedSemaphore"}
_ASYNCIO_LOCK_NAMES = {"Lock", "Semaphore", "BoundedSemaphore"}


def _collect_asyncio_primitive_types() -> tuple[type, ...]:
    """Return concrete asyncio synchronisation primitive types."""

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


def _assign_dataclass_attr(target: object, name: str, value: object) -> bool:
    """Assign ``value`` to ``target.name`` bypassing frozen guards when possible."""

    try:
        setattr(target, name, value)
        return True
    except Exception:
        try:
            object.__setattr__(target, name, value)
        except Exception:
            return False
        return True


def _recreate_dataclass_if_needed(
    obj: object, mutated_fields: dict[str, object]
) -> object:
    """Return a dataclass instance reflecting ``mutated_fields``.

    ``dataclasses.replace`` handles frozen/slots dataclasses while preserving
    ``eq``/``hash`` semantics. When ``replace`` is not viable (``init=False``
    fields, custom ``__init__``), we fall back to in-place assignment using
    :func:`_assign_dataclass_attr`.
    """

    if not mutated_fields:
        return obj

    try:
        return replace(obj, **mutated_fields)
    except Exception:
        for name, value in mutated_fields.items():
            _assign_dataclass_attr(obj, name, value)
        return obj


def _scan(obj: object, seen: set[int], loop: asyncio.AbstractEventLoop) -> object:
    obj_id = id(obj)
    if obj_id in seen:
        return obj
    seen.add(obj_id)

    if _is_asyncio_primitive(obj):
        return _maybe_recreate_lock(obj, loop)

    if isinstance(obj, Mapping):
        mutated = False
        updates: list[tuple[object, object]] = []
        for key, value in list(obj.items()):
            new_value = _scan(value, seen, loop)
            mutated = mutated or new_value is not value
            updates.append((key, new_value))
        if not mutated:
            return obj

        if isinstance(obj, MutableMapping):
            for key, value in updates:
                obj[key] = value
            return obj

        if isinstance(obj, MappingProxyType):
            return MappingProxyType(dict(updates))

        try:
            return type(obj)(updates)
        except Exception:
            return dict(updates)

    if isinstance(obj, list):
        for idx, value in enumerate(list(obj)):
            new_value = _scan(value, seen, loop)
            if new_value is not value:
                obj[idx] = new_value
        return obj

    if isinstance(obj, tuple):
        mutated = False
        new_items: list[object] = []
        for value in obj:
            new_value = _scan(value, seen, loop)
            mutated = mutated or new_value is not value
            new_items.append(new_value)
        if mutated:
            return tuple(new_items)
        return obj

    if isinstance(obj, set):
        mutated = False
        new_values: list[object] = []
        for value in list(obj):
            new_value = _scan(value, seen, loop)
            mutated = mutated or new_value is not value
            new_values.append(new_value)
        if mutated:
            obj.clear()
            obj.update(new_values)
        return obj

    if isinstance(obj, frozenset):
        mutated = False
        new_values: list[object] = []
        for value in obj:
            new_value = _scan(value, seen, loop)
            mutated = mutated or new_value is not value
            new_values.append(new_value)
        if mutated:
            return type(obj)(new_values)
        return obj

    if is_dataclass(obj) and not isinstance(obj, type):
        replacements: dict[str, object] = {}
        for field in fields(obj):
            try:
                value = getattr(obj, field.name)
            except AttributeError:
                continue
            new_value = _scan(value, seen, loop)
            if new_value is not value:
                replacements[field.name] = new_value

        if replacements:
            obj = _recreate_dataclass_if_needed(obj, replacements)

        try:
            extra_attrs = vars(obj)
        except TypeError:
            extra_attrs = None
        if extra_attrs:
            for name, value in list(extra_attrs.items()):
                new_value = _scan(value, seen, loop)
                if new_value is not value:
                    _assign_dataclass_attr(obj, name, new_value)
        return obj

    if isinstance(obj, SimpleNamespace):
        for name, value in list(vars(obj).items()):
            new_value = _scan(value, seen, loop)
            if new_value is not value:
                setattr(obj, name, new_value)
        return obj

    slot_names = getattr(type(obj), "__slots__", ())
    if slot_names:
        slot_tuple = (slot_names,) if isinstance(slot_names, str) else slot_names
        for name in slot_tuple:
            if hasattr(obj, name):
                value = getattr(obj, name)
                new_value = _scan(value, seen, loop)
                if new_value is not value:
                    try:
                        setattr(obj, name, new_value)
                    except Exception:
                        pass
        return obj

    module_name = getattr(obj.__class__, "__module__", "")
    if hasattr(obj, "__dict__") and not isinstance(obj, ModuleType) and module_name.startswith(
        ("ai_trading", "tests", "__main__")
    ):
        for name, value in list(vars(obj).items()):
            new_value = _scan(value, seen, loop)
            if new_value is not value:
                try:
                    setattr(obj, name, new_value)
                except Exception:
                    pass
        return obj

    return obj


def _rebind_worker_closure(worker: Callable[[str], Awaitable[T]], loop: asyncio.AbstractEventLoop) -> None:
    """Rebind foreign-loop locks captured in ``worker``'s closure to ``loop``."""

    cells = getattr(worker, "__closure__", None)
    if not cells:
        return
    seen: set[int] = set()
    for cell in cells:
        try:
            original = cell.cell_contents
        except ValueError:
            continue
        new_value = _scan(original, seen, loop)
        if new_value is not original:
            cell.cell_contents = new_value


SUCCESSFUL_SYMBOLS: set[str] = set()
FAILED_SYMBOLS: set[str] = set()
PEAK_SIMULTANEOUS_WORKERS: int = 0


async def run_with_concurrency(
    symbols: Iterable[str],
    worker: Callable[[str], Awaitable[T]],
    *,
    max_concurrency: int = 4,
    timeout_s: float | None = None,
) -> tuple[dict[str, T | None], set[str], set[str]]:
    """Execute ``worker`` for each symbol with at most ``max_concurrency`` tasks in flight."""

    global PEAK_SIMULTANEOUS_WORKERS

    PEAK_SIMULTANEOUS_WORKERS = 0
    SUCCESSFUL_SYMBOLS.clear()
    FAILED_SYMBOLS.clear()

    loop = asyncio.get_running_loop()
    _rebind_worker_closure(worker, loop)

    limit = max(1, int(max_concurrency))
    host_limit = _get_effective_host_limit()
    if host_limit is not None:
        limit = min(limit, host_limit)

    results: dict[str, T | None] = {}
    pending_tasks: set[asyncio.Task[tuple[str, T | BaseException]]] = set()
    symbol_iter = iter(symbols)

    async def _execute(symbol: str) -> tuple[str, T | BaseException]:
        try:
            result = await worker(symbol)
        except BaseException as exc:  # noqa: BLE001 - captured for aggregation
            return symbol, exc
        return symbol, result

    async def _drive() -> None:
        nonlocal pending_tasks
        global PEAK_SIMULTANEOUS_WORKERS

        symbols_exhausted = False

        while True:
            while len(pending_tasks) < limit and not symbols_exhausted:
                try:
                    symbol = next(symbol_iter)
                except StopIteration:
                    symbols_exhausted = True
                    break
                task = asyncio.create_task(_execute(symbol))
                pending_tasks.add(task)
                current = len(pending_tasks)
                if current > PEAK_SIMULTANEOUS_WORKERS:
                    PEAK_SIMULTANEOUS_WORKERS = current

            if not pending_tasks:
                if symbols_exhausted:
                    break
                continue

            done, still_pending = await asyncio.wait(
                pending_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            pending_tasks = set(still_pending)

            for task in done:
                try:
                    symbol, outcome = task.result()
                except BaseException:
                    continue
                if isinstance(outcome, BaseException):
                    FAILED_SYMBOLS.add(symbol)
                    results[symbol] = None
                else:
                    SUCCESSFUL_SYMBOLS.add(symbol)
                    results[symbol] = outcome

            if pending_tasks:
                continue

    driver = _drive()

    try:
        await asyncio.wait_for(driver, timeout_s) if timeout_s is not None else await driver
    except asyncio.TimeoutError:
        for task in list(pending_tasks):
            task.cancel()
        gathered = await asyncio.gather(*pending_tasks, return_exceptions=True)
        for entry in gathered:
            if isinstance(entry, tuple) and len(entry) == 2:
                symbol, outcome = entry
                if isinstance(outcome, BaseException):
                    FAILED_SYMBOLS.add(symbol)
                    results.setdefault(symbol, None)
                else:
                    SUCCESSFUL_SYMBOLS.add(symbol)
                    results.setdefault(symbol, outcome)

    return results, SUCCESSFUL_SYMBOLS.copy(), FAILED_SYMBOLS.copy()


async def run_with_concurrency_limit(
    symbols: Iterable[str],
    worker: Callable[[str], Awaitable[T]],
    *,
    max_concurrency: int = 4,
    timeout_s: float | None = None,
) -> tuple[dict[str, T | None], set[str], set[str]]:
    """Compatibility alias for ``run_with_concurrency``."""

    return await run_with_concurrency(
        symbols,
        worker,
        max_concurrency=max_concurrency,
        timeout_s=timeout_s,
    )


__all__ = [
    "run_with_concurrency",
    "run_with_concurrency_limit",
    "SUCCESSFUL_SYMBOLS",
    "FAILED_SYMBOLS",
    "PEAK_SIMULTANEOUS_WORKERS",
]

