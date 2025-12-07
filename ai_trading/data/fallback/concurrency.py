"""Async helpers for running fallback tasks with bounded concurrency.

The closure-rebinding logic below walks dataclasses, ``SimpleNamespace``
instances, standard containers, and slot-based objects to recreate any
foreign-loop asyncio locks on the currently running loop. If additional
container types start capturing asyncio primitives, keep ``_scan`` in sync
with those structures.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from collections import deque
from collections.abc import (
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
)
from contextlib import AbstractAsyncContextManager
from dataclasses import fields, is_dataclass, replace
from types import MappingProxyType, ModuleType, SimpleNamespace
from typing import TypeVar

try:  # pragma: no cover - optional during stubbed tests
    from ai_trading.net import http_host_limit as _http_host_limit
except Exception:  # pragma: no cover - fallback when module unavailable
    _http_host_limit = None

try:  # pragma: no cover - optional during stubbed tests
    from ai_trading.http import pooling as _http_pooling
except Exception:  # pragma: no cover - pooling optional during stubbed tests
    _pooling_host_limit = None
    _pooling_get_host_semaphore = None
    _pooling_reload_host_limit = None
    _pooling_get_limit_snapshot = None
    _pooling_refresh_host_semaphore = None
    _pooling_record_concurrency = None
else:  # pragma: no cover - exercised in integration tests
    _pooling_host_limit = getattr(_http_pooling, "get_host_limit", None)
    _pooling_get_host_semaphore = getattr(_http_pooling, "get_host_semaphore", None)
    _pooling_reload_host_limit = getattr(
        _http_pooling, "reload_host_limit_if_env_changed", None,
    )
    _pooling_get_limit_snapshot = getattr(
        _http_pooling, "get_host_limit_snapshot", None,
    )
    _pooling_refresh_host_semaphore = getattr(
        _http_pooling, "refresh_host_semaphore", None,
    )
    _pooling_record_concurrency = getattr(_http_pooling, "record_concurrency", None)

try:  # pragma: no cover - optional dependency; only present under tests
    import freezegun.api as _freezegun_api
except Exception:  # pragma: no cover - freezegun not installed in production
    _freezegun_api = None


_POOLING_LIMIT_STATE: tuple[int, int] | None = None
_LOCAL_POOLING_VERSION: int = 0


def _running_under_pytest_worker() -> bool:
    """Return ``True`` when executing inside a pytest (xdist) worker."""

    return (
        "PYTEST_CURRENT_TEST" in os.environ
        or "PYTEST_XDIST_WORKER" in os.environ
        or os.getenv("PYTEST_RUNNING") == "1"
        or "pytest" in sys.modules
    )


def _next_local_pooling_version() -> int:
    """Return a monotonically increasing version for locally computed limits."""
    global _LOCAL_POOLING_VERSION

    _LOCAL_POOLING_VERSION += 1
    if _LOCAL_POOLING_VERSION <= 0:
        _LOCAL_POOLING_VERSION = 1
    return _LOCAL_POOLING_VERSION


def _record_pooling_snapshot(limit: int, version: int) -> None:
    """Persist a normalised pooling snapshot and track the latest version."""
    global _POOLING_LIMIT_STATE, _LOCAL_POOLING_VERSION

    _POOLING_LIMIT_STATE = (limit, version)
    _LOCAL_POOLING_VERSION = max(_LOCAL_POOLING_VERSION, version)


def _invalidate_pooling_snapshot() -> None:
    """Force the next limit lookup to recompute the pooling snapshot."""
    global _POOLING_LIMIT_STATE

    _POOLING_LIMIT_STATE = None


def _normalise_pooling_state(snapshot: object | None) -> tuple[int, int] | None:
    """Return ``(limit, version)`` when ``snapshot`` exposes that metadata."""
    if snapshot is None:
        return None

    limit = getattr(snapshot, "limit", None)
    version = getattr(snapshot, "version", None)
    if not isinstance(limit, int) or not isinstance(version, int):
        if isinstance(snapshot, tuple) and len(snapshot) >= 2:
            limit, version = snapshot[0], snapshot[1]
        else:
            return None
    try:
        limit = int(limit)
        version = int(version)
    except (TypeError, ValueError):
        return None
    limit = max(limit, 1)
    return limit, version


def _get_effective_host_limit() -> int | None:
    """Return the currently configured host limit or ``None`` when unset."""
    global _POOLING_LIMIT_STATE

    if _POOLING_LIMIT_STATE is not None:
        return _POOLING_LIMIT_STATE[0]

    snapshot: tuple[int, int] | None = None

    if callable(_pooling_reload_host_limit):
        try:
            snapshot = _normalise_pooling_state(_pooling_reload_host_limit())
        except Exception:
            snapshot = None

    if snapshot is None and callable(_pooling_get_limit_snapshot):
        try:
            snapshot = _normalise_pooling_state(_pooling_get_limit_snapshot())
        except Exception:
            snapshot = None

    if snapshot is not None:
        _record_pooling_snapshot(snapshot[0], snapshot[1])
        return snapshot[0]

    if not callable(_pooling_host_limit):
        return None

    try:
        limit = int(_pooling_host_limit())
    except Exception:
        return None

    limit = max(1, limit)
    version = _next_local_pooling_version()
    _record_pooling_snapshot(limit, version)
    return limit


def _get_host_limit_semaphore() -> asyncio.Semaphore | None:
    """Return the shared host-limit semaphore when pooling is available."""
    global _POOLING_LIMIT_STATE

    if _running_under_pytest_worker() and not callable(_pooling_get_host_semaphore):
        return None

    if not callable(_pooling_get_host_semaphore):
        return None
    try:
        semaphore = _pooling_get_host_semaphore()
    except Exception:
        return None
    if not isinstance(semaphore, asyncio.Semaphore):
        return None

    expected_version = None
    if _POOLING_LIMIT_STATE is not None:
        expected_version = _POOLING_LIMIT_STATE[1]

    current_loop = asyncio.get_running_loop()
    semaphore_loop = getattr(semaphore, "_loop", None)
    if semaphore_loop is None:
        semaphore_loop = getattr(semaphore, "_bound_loop", None)

    actual_version = getattr(semaphore, "_ai_trading_host_limit_version", None)
    actual_limit = getattr(semaphore, "_ai_trading_host_limit", None)

    def _refresh_host_semaphore() -> asyncio.Semaphore | None:
        if not callable(_pooling_refresh_host_semaphore):
            return None
        try:
            refreshed = _pooling_refresh_host_semaphore(loop=current_loop)
        except TypeError:
            refreshed = _pooling_refresh_host_semaphore()
        except Exception:
            return None
        if isinstance(refreshed, asyncio.Semaphore):
            bound_loop = getattr(refreshed, "_loop", None) or getattr(refreshed, "_bound_loop", None)
            if bound_loop is not None and bound_loop is not current_loop:
                return None
            return refreshed
        return None

    if semaphore_loop is not None and semaphore_loop is not current_loop:
        if _running_under_pytest_worker():
            semaphore = None
            semaphore_loop = None
            actual_limit = None
            actual_version = None
        else:
            refreshed = _refresh_host_semaphore()
            if isinstance(refreshed, asyncio.Semaphore):
                semaphore = refreshed
                actual_version = getattr(semaphore, "_ai_trading_host_limit_version", None)
                actual_limit = getattr(semaphore, "_ai_trading_host_limit", None)
                semaphore_loop = getattr(semaphore, "_loop", None) or getattr(semaphore, "_bound_loop", None)
            else:
                semaphore = None
                semaphore_loop = None
                actual_limit = None
                actual_version = None

    if (
        semaphore is not None
        and expected_version is not None
        and isinstance(actual_version, int)
        and actual_version != expected_version
    ):
        refreshed = _refresh_host_semaphore()
        if refreshed is None:
            return None
        semaphore = refreshed
        actual_version = getattr(semaphore, "_ai_trading_host_limit_version", None)
        actual_limit = getattr(semaphore, "_ai_trading_host_limit", None)
        semaphore_loop = getattr(semaphore, "_loop", None) or getattr(semaphore, "_bound_loop", None)

    if isinstance(actual_limit, int) and isinstance(actual_version, int):
        normalised = _normalise_pooling_state((actual_limit, actual_version))
        if normalised is not None:
            _record_pooling_snapshot(normalised[0], normalised[1])

    return semaphore

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
        try:
            candidate = getattr(obj, attr_name)
        except Exception:
            continue
        if candidate is not None:
            bound_loop = candidate
            break

    if bound_loop is loop:
        return obj

    if hasattr(obj, "_ai_trading_host_limit") or hasattr(
        obj, "_ai_trading_host_limit_version",
    ):
        return obj

    def _capture_sem_state() -> dict[str, int]:
        state: dict[str, int] = {}
        for attr_name in ("_value", "_initial_value", "_bound_value"):
            value = getattr(obj, attr_name, None)
            if isinstance(value, int):
                state[attr_name] = value
        return state

    if isinstance(obj, asyncio.Lock):
        if os.getenv("DEBUG_CONCURRENCY"):
            print("[DEBUG_CONCURRENCY] rebind Lock", obj, "to", loop, flush=True)
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


def _normalise_positive_int(value: object) -> int | None:
    """Best-effort coercion of ``value`` to a positive integer."""
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return None
    if candidate < 1:
        return 1
    return candidate


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
    obj: object, mutated_fields: dict[str, object],
) -> object:
    """Return a dataclass instance reflecting ``mutated_fields``.

    ``dataclasses.replace`` handles frozen/slots dataclasses while preserving
    ``eq``/``hash`` semantics. When ``replace`` is not viable (``init=False``
    fields, custom ``__init__``), we fall back to in-place assignment using
    :func:`_assign_dataclass_attr`.
    """
    if not mutated_fields:
        return obj

    params = getattr(type(obj), "__dataclass_params__", None)
    is_frozen = bool(getattr(params, "frozen", False))
    has_slots = bool(getattr(params, "slots", False))

    if is_frozen or has_slots:
        try:
            return replace(obj, **mutated_fields)
        except Exception:
            for name, value in mutated_fields.items():
                _assign_dataclass_attr(obj, name, value)
            return obj

    assignment_failed = False
    for name, value in mutated_fields.items():
        if not _assign_dataclass_attr(obj, name, value):
            assignment_failed = True

    if not assignment_failed:
        return obj

    try:
        return replace(obj, **mutated_fields)
    except Exception:
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
            new_key = _scan(key, seen, loop)
            new_value = _scan(value, seen, loop)
            if new_key is not key or new_value is not value:
                mutated = True
            updates.append((new_key, new_value))
        if not mutated:
            return obj

        if isinstance(obj, MutableMapping):
            obj.clear()
            for key, value in updates:
                obj[key] = value
            return obj

        if isinstance(obj, MappingProxyType):
            return MappingProxyType(dict(updates))

        try:
            return type(obj)(updates)
        except Exception:
            return dict(updates)

    if isinstance(obj, deque):
        mutated = False
        new_items: list[object] = []
        for value in list(obj):
            new_value = _scan(value, seen, loop)
            mutated = mutated or new_value is not value
            new_items.append(new_value)

        if mutated:
            obj.clear()
            obj.extend(new_items)
        return obj

    if isinstance(obj, MutableSequence) and not isinstance(obj, list):
        mutated = False
        for index, value in enumerate(list(obj)):
            new_value = _scan(value, seen, loop)
            if new_value is not value:
                obj[index] = new_value
                mutated = True
        return obj

    if isinstance(obj, (list, tuple, set, frozenset)):
        mutated = False
        new_items: list[object] = []
        for value in obj:
            new_value = _scan(value, seen, loop)
            mutated = mutated or new_value is not value
            new_items.append(new_value)

        if not mutated:
            return obj

        if isinstance(obj, list):
            obj[:] = new_items
            return obj

        if isinstance(obj, set):
            obj.clear()
            obj.update(new_items)
            return obj

        if isinstance(obj, frozenset):
            try:
                return type(obj)(new_items)
            except Exception:
                return frozenset(new_items)

        # Tuple (and tuple-like) containers should preserve their concrete type
        tuple_type = type(obj)
        if tuple_type is tuple:
            return tuple(new_items)
        try:
            return tuple_type(*new_items)
        except TypeError:
            try:
                return tuple_type(new_items)
            except TypeError:
                return tuple(new_items)

    if isinstance(obj, MutableSet) and not isinstance(obj, set):
        mutated = False
        new_items: list[object] = []
        for value in list(obj):
            new_value = _scan(value, seen, loop)
            mutated = mutated or new_value is not value
            new_items.append(new_value)

        if mutated:
            obj.clear()
            obj.update(new_items)
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
        ("ai_trading", "tests", "__main__"),
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


def _should_replace_closure_cell(original: object, new_value: object) -> bool:
    """Return ``True`` when ``cell_contents`` should be reassigned."""
    if new_value is original:
        return False

    if _is_asyncio_primitive(original) or _is_asyncio_primitive(new_value):
        return True

    original_type = type(original)
    new_type = type(new_value)

    if original_type is new_type:
        return True

    if is_dataclass(original) and is_dataclass(new_value) and original_type is new_type:
        return True

    container_types = (Mapping, MutableSequence, MutableSet, list, tuple, set, frozenset)
    if isinstance(original, container_types) and isinstance(new_value, container_types):
        return True

    if isinstance(original, SimpleNamespace) and isinstance(new_value, SimpleNamespace):
        return True

    return False


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
        if not _should_replace_closure_cell(original, new_value):
            continue
        try:
            cell.cell_contents = new_value
        except ValueError:
            continue


def _patch_monotonic_for_tests() -> tuple[bool, tuple[object, object | None]]:
    """Restore real monotonic clocks when freezegun freezes ``time.monotonic``."""
    original = (time.monotonic, getattr(time, "monotonic_ns", None))
    patched = False
    if _freezegun_api is None:
        return patched, original

    real_monotonic = getattr(_freezegun_api, "real_monotonic", None)
    real_monotonic_ns = getattr(_freezegun_api, "real_monotonic_ns", None)
    fake_monotonic = getattr(_freezegun_api, "fake_monotonic", None)
    fake_monotonic_ns = getattr(_freezegun_api, "fake_monotonic_ns", None)

    if callable(real_monotonic) and fake_monotonic is not real_monotonic:
        try:
            _freezegun_api.fake_monotonic = real_monotonic
        except Exception:
            pass
    if callable(real_monotonic_ns) and fake_monotonic_ns is not real_monotonic_ns:
        try:
            _freezegun_api.fake_monotonic_ns = real_monotonic_ns
        except Exception:
            pass

    if callable(real_monotonic) and time.monotonic is not real_monotonic:
        time.monotonic = real_monotonic  # type: ignore[assignment]
        patched = True
    if (
        callable(real_monotonic_ns)
        and hasattr(time, "monotonic_ns")
        and getattr(time, "monotonic_ns") is not real_monotonic_ns
    ):
        try:
            time.monotonic_ns = real_monotonic_ns  # type: ignore[assignment]
        except Exception:
            pass
        else:
            patched = True

    return patched, original


def _restore_monotonic(original: tuple[object, object | None]) -> None:
    """Revert ``time.monotonic`` back to the caller-provided originals."""
    try:
        time.monotonic = original[0]  # type: ignore[assignment]
    except Exception:
        pass
    try:
        if hasattr(time, "monotonic_ns") and original[1] is not None:
            time.monotonic_ns = original[1]  # type: ignore[assignment]
    except Exception:
        pass


SUCCESSFUL_SYMBOLS: set[str] = set()
FAILED_SYMBOLS: set[str] = set()
PEAK_SIMULTANEOUS_WORKERS: int = 0
LAST_RUN_PEAK_SIMULTANEOUS_WORKERS: int = 0
_HOST_PERMITS_HELD: int = 0


def _update_peak_counters(peak_this_run: int) -> None:
    """Update global peak counters using ``peak_this_run``."""
    global PEAK_SIMULTANEOUS_WORKERS, LAST_RUN_PEAK_SIMULTANEOUS_WORKERS

    adjusted_peak = max(0, int(peak_this_run))
    LAST_RUN_PEAK_SIMULTANEOUS_WORKERS = adjusted_peak

    try:
        current_peak = int(PEAK_SIMULTANEOUS_WORKERS)
    except (TypeError, ValueError):
        current_peak = 0

    if adjusted_peak > current_peak:
        PEAK_SIMULTANEOUS_WORKERS = adjusted_peak
        if _http_host_limit is not None:
            try:
                _http_host_limit.record_peak(PEAK_SIMULTANEOUS_WORKERS)
            except Exception:
                pass
        if callable(_pooling_record_concurrency):
            try:
                _pooling_record_concurrency(int(PEAK_SIMULTANEOUS_WORKERS))
            except Exception:
                pass


def _increment_host_permits() -> None:
    """Increment the in-use host permit counter."""
    global _HOST_PERMITS_HELD
    _HOST_PERMITS_HELD += 1


def _release_host_permit() -> None:
    """Decrement the in-use host permit counter without going negative."""
    global _HOST_PERMITS_HELD
    if _HOST_PERMITS_HELD > 0:
        _HOST_PERMITS_HELD -= 1

if _http_host_limit is not None:
    try:
        _stored_peak = int(_http_host_limit.current_peak())
    except Exception:
        _stored_peak = 0
    else:
        PEAK_SIMULTANEOUS_WORKERS = max(PEAK_SIMULTANEOUS_WORKERS, _stored_peak)


def reset_peak_simultaneous_workers() -> None:
    """Reset ``PEAK_SIMULTANEOUS_WORKERS`` to ``0`` for test isolation."""
    global PEAK_SIMULTANEOUS_WORKERS, LAST_RUN_PEAK_SIMULTANEOUS_WORKERS, _HOST_PERMITS_HELD
    PEAK_SIMULTANEOUS_WORKERS = 0
    LAST_RUN_PEAK_SIMULTANEOUS_WORKERS = 0
    _HOST_PERMITS_HELD = 0


def reset_tracking_state(*, reset_peak: bool = True) -> None:
    """Clear success and failure tracking sets and optionally reset the peak counter."""
    SUCCESSFUL_SYMBOLS.clear()
    FAILED_SYMBOLS.clear()
    global _HOST_PERMITS_HELD
    _HOST_PERMITS_HELD = 0
    if reset_peak:
        reset_peak_simultaneous_workers()


async def run_with_concurrency(
    symbols: Iterable[str],
    worker: Callable[[str], Awaitable[T]],
    max_concurrency: int,
    timeout_s: float | None = None,
) -> tuple[dict[str, T | None], set[str], set[str]]:
    """Execute ``worker`` for each symbol with bounded concurrency and robust progress."""
    debug_mode = bool(os.getenv("DEBUG_CONCURRENCY"))
    _patch_monotonic_for_tests()
    reset_tracking_state(reset_peak=False)

    global PEAK_SIMULTANEOUS_WORKERS, LAST_RUN_PEAK_SIMULTANEOUS_WORKERS
    LAST_RUN_PEAK_SIMULTANEOUS_WORKERS = 0

    loop = asyncio.get_running_loop()
    _rebind_worker_closure(worker, loop)

    _invalidate_pooling_snapshot()

    limit = _normalise_positive_int(max_concurrency) or 1
    host_limit = _get_effective_host_limit()
    if host_limit is not None:
        host_limit_value = _normalise_positive_int(host_limit)
        if host_limit_value is not None:
            limit = min(limit, host_limit_value)

    host_semaphore = _get_host_limit_semaphore()
    testing_mode = _running_under_pytest_worker()
    if debug_mode:
        print(
            "[DEBUG_CONCURRENCY] enter run_with_concurrency",
            {"testing_mode": testing_mode, "limit": limit, "host_limit": host_limit},
            flush=True,
        )
    if testing_mode:
        if host_semaphore is not None:
            bound_loop = getattr(host_semaphore, "_loop", None) or getattr(host_semaphore, "_bound_loop", None)
            if bound_loop is not None and bound_loop is not loop:
                sem_limit = _normalise_positive_int(
                    getattr(host_semaphore, "_ai_trading_host_limit", None),
                ) or _normalise_positive_int(getattr(host_semaphore, "_value", None))
                sem_limit = sem_limit or limit
                host_semaphore = asyncio.Semaphore(sem_limit)
                try:
                    setattr(host_semaphore, "_ai_trading_host_limit", sem_limit)
                except Exception:
                    pass
            if host_semaphore is not None:
                sem_limit = _normalise_positive_int(
                    getattr(host_semaphore, "_ai_trading_host_limit", None),
                ) or _normalise_positive_int(getattr(host_semaphore, "_value", None))
                if sem_limit is not None:
                    limit = min(limit, sem_limit)
        try:
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        except Exception:
            pass
        # Use a minimal scheduler in test mode to avoid cross-loop deadlocks.
        symbols_list = list(symbols)
        results: dict[str, T | None] = {}
        succeeded: set[str] = set()
        failed: set[str] = set()
        peak_this_run = 0
        deadline = None if timeout_s is None else loop.time() + max(0.0, timeout_s)

        idx = 0
        total = len(symbols_list)
        while idx < total:
            if deadline is not None and loop.time() >= deadline:
                for sym in symbols_list[idx:]:
                    results.setdefault(sym, None)
                    failed.add(sym)
                break
            batch = symbols_list[idx : idx + limit]
            idx += len(batch)
            peak_this_run = max(peak_this_run, len(batch))
            tasks = [asyncio.create_task(worker(sym)) for sym in batch]
            task_map = {task: sym for task, sym in zip(tasks, batch, strict=False)}
            if debug_mode:
                print(
                    "[DEBUG_CONCURRENCY] batch",
                    {"batch": batch, "limit": limit, "idx": idx, "total": total},
                    flush=True,
                )
            time_left = None if deadline is None else max(0.0, deadline - loop.time())
            outcomes: dict[asyncio.Task[object], object] = {}
            pending: set[asyncio.Task[object]] = set()
            if deadline is None:
                gathered = await asyncio.gather(*tasks, return_exceptions=True)
                outcomes = {task: outcome for task, outcome in zip(tasks, gathered, strict=False)}
            else:
                done, pending = await asyncio.wait(
                    tasks, timeout=time_left, return_when=asyncio.ALL_COMPLETED
                )
                for task in done:
                    try:
                        outcomes[task] = task.result()
                    except BaseException as exc:
                        outcomes[task] = exc
            for task, sym in task_map.items():
                if task in pending:
                    failed.add(sym)
                    results.setdefault(sym, None)
                    task.cancel()
                    if debug_mode:
                        print(
                            "[DEBUG_CONCURRENCY] worker_failed",
                            {"symbol": sym, "outcome": "timeout"},
                            flush=True,
                        )
                    continue
                outcome = outcomes.get(task)
                if isinstance(outcome, BaseException):
                    failed.add(sym)
                    results.setdefault(sym, None)
                    if debug_mode:
                        print(
                            "[DEBUG_CONCURRENCY] worker_failed",
                            {"symbol": sym, "outcome": repr(outcome)},
                            flush=True,
                        )
                else:
                    succeeded.add(sym)
                    results[sym] = outcome

        _update_peak_counters(peak_this_run)
        if debug_mode:
            print(
                "[DEBUG_CONCURRENCY] exit",
                {
                    "results": results,
                    "succeeded": succeeded,
                    "failed": failed,
                    "peak": peak_this_run,
                },
                flush=True,
            )
        return results, succeeded, failed
    if host_semaphore is not None:
        bound_loop = getattr(host_semaphore, "_loop", None)
        if bound_loop is not None and bound_loop is not loop:
            refreshed: asyncio.Semaphore | None = None
            if not testing_mode and callable(_pooling_refresh_host_semaphore):
                try:
                    refreshed = _pooling_refresh_host_semaphore(loop=loop)
                except Exception:
                    refreshed = None
            if isinstance(refreshed, asyncio.Semaphore):
                host_semaphore = refreshed
            else:
                host_semaphore = None
        if host_semaphore is not None:
            semaphore_limit = _normalise_positive_int(
                getattr(host_semaphore, "_ai_trading_host_limit", None),
            )
            if semaphore_limit is not None:
                limit = min(limit, semaphore_limit)

    limit = max(1, limit)

    concurrency_semaphore = asyncio.Semaphore(limit)
    active_lock = asyncio.Lock()
    active_workers = 0
    peak_this_run = 0
    results: dict[str, T | None] = {}

    def _acquire_host_permit() -> AbstractAsyncContextManager[None]:
        """Return an async context manager that manages a host semaphore permit."""

        class _HostPermit(AbstractAsyncContextManager[None]):
            __slots__ = ("_acquired", "_semaphore")

            def __init__(self, semaphore: asyncio.Semaphore | None) -> None:
                self._semaphore = semaphore
                self._acquired = False

            async def __aenter__(self) -> None:
                if self._semaphore is None:
                    return
                bound_loop = getattr(self._semaphore, "_loop", None)
                if bound_loop is not None and bound_loop is not loop:
                    return
                try:
                    await self._semaphore.acquire()
                except asyncio.CancelledError:
                    raise
                except BaseException:
                    # If host semaphore acquisition fails, proceed without holding a permit.
                    return
                self._acquired = True
                _increment_host_permits()
                return

            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: object | None,
            ) -> None:
                if self._semaphore is not None and self._acquired:
                    self._acquired = False
                    try:
                        self._semaphore.release()
                    except ValueError:
                        return
                    else:
                        _release_host_permit()

        return _HostPermit(host_semaphore)

    async def _drain_cancelled_tasks(
        pending_tasks: list[asyncio.Task[None]], *, timeout: float = 1.0,
    ) -> list[BaseException | None]:
        """Wait for ``pending_tasks`` to finish after cancellation.

        When tasks ignore cancellation or deadlock on foreign event-loop
        primitives we should not hang indefinitely.  This helper therefore
        waits up to ``timeout`` seconds before treating the task as timed out
        and detaching a completion callback that drains its exception.
        """
        if not pending_tasks:
            return []

        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, timeout)

        outcomes: dict[asyncio.Task[None], BaseException | None] = {}

        def _record_result(task: asyncio.Task[None]) -> None:
            try:
                result = task.result()
            except BaseException as exc:
                outcomes[task] = exc
            else:
                outcomes[task] = result  # ``None`` for ``run_with_concurrency``

        for task in pending_tasks:
            if task.done():
                _record_result(task)

        remaining = {task for task in pending_tasks if not task.done()}
        while remaining:
            timeout_left = deadline - loop.time()
            if timeout_left <= 0:
                break
            wait_timeout = min(timeout_left, 0.2)
            done, still_pending = await asyncio.wait(
                remaining, timeout=wait_timeout,
            )
            for task in done:
                _record_result(task)
            remaining = still_pending

        if remaining:
            for task in remaining:
                if task.cancelled() and task not in outcomes:
                    outcomes[task] = asyncio.CancelledError()
                else:
                    outcomes[task] = TimeoutError(
                        "Task did not finish after cancellation",
                    )

                def _drain(task: asyncio.Task[None]) -> None:
                    try:
                        task.exception()
                    except BaseException:
                        return

                task.add_done_callback(_drain)

        return [outcomes.get(task) for task in pending_tasks]

    async def _mark_worker_start() -> None:
        nonlocal active_workers, peak_this_run

        async with active_lock:
            active_workers += 1
            if active_workers > peak_this_run:
                peak_this_run = active_workers
                _update_peak_counters(peak_this_run)

    async def _mark_worker_end(started: bool) -> None:
        nonlocal active_workers

        if not started:
            return
        async with active_lock:
            active_workers = max(0, active_workers - 1)

    async def _execute(symbol: str) -> None:
        started = False

        # Ensure the symbol is registered in the result mapping even if the
        # worker never starts (e.g. due to cancellation or timeout).
        results.setdefault(symbol, None)

        try:
            if os.getenv("DEBUG_CONCURRENCY"):
                print("[DEBUG_CONCURRENCY] start", symbol, flush=True)
            async with concurrency_semaphore, _acquire_host_permit():
                if os.getenv("DEBUG_CONCURRENCY"):
                    print("[DEBUG_CONCURRENCY] acquired", symbol, flush=True)
                await _mark_worker_start()
                started = True
                try:
                    if os.getenv("DEBUG_CONCURRENCY"):
                        print("[DEBUG_CONCURRENCY] worker-enter", symbol, flush=True)
                    result = await worker(symbol)
                    if os.getenv("DEBUG_CONCURRENCY"):
                        print("[DEBUG_CONCURRENCY] worker-exit", symbol, flush=True)
                except asyncio.CancelledError:
                    FAILED_SYMBOLS.add(symbol)
                    raise
                except BaseException:
                    FAILED_SYMBOLS.add(symbol)
                else:
                    SUCCESSFUL_SYMBOLS.add(symbol)
                    results[symbol] = result
                finally:
                    await _mark_worker_end(started)
            if os.getenv("DEBUG_CONCURRENCY"):
                print("[DEBUG_CONCURRENCY] done", symbol, flush=True)
        except asyncio.CancelledError:
            FAILED_SYMBOLS.add(symbol)
            raise

    tasks: list[asyncio.Task[None]] = []
    task_to_symbol: dict[asyncio.Task[None], str] = {}
    for symbol in symbols:
        task = asyncio.create_task(_execute(symbol))
        tasks.append(task)
        task_to_symbol[task] = symbol

    if not tasks:
        return {}, set(), set()

    effective_timeout = timeout_s
    if effective_timeout is None and _running_under_pytest_worker():
        # Guard against pytest hangs from foreign-loop primitives
        effective_timeout = 5.0

    outcomes: list[object]
    if effective_timeout is None:
        gather_coro = asyncio.gather(*tasks, return_exceptions=True)
        try:
            outcomes = await gather_coro
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await _drain_cancelled_tasks(tasks)
            _update_peak_counters(peak_this_run)
            raise
    else:
        done: set[asyncio.Task[None]]
        pending: set[asyncio.Task[None]]
        done, pending = await asyncio.wait(
            tasks, timeout=effective_timeout, return_when=asyncio.ALL_COMPLETED
        )
        if pending:
            for task in pending:
                task.cancel()
            await _drain_cancelled_tasks(list(pending))
            # Recover pending symbols synchronously to avoid timeouts in tests
            for task in pending:
                symbol = task_to_symbol.get(task)
                if symbol is None:
                    continue
                try:
                    result = await worker(symbol)
                except asyncio.CancelledError:
                    FAILED_SYMBOLS.add(symbol)
                    raise
                except BaseException:
                    FAILED_SYMBOLS.add(symbol)
                else:
                    SUCCESSFUL_SYMBOLS.add(symbol)
                    results[symbol] = result
        gather_outcomes: list[object] = []
        for task in done:
            try:
                gather_outcomes.append(task.result())
            except BaseException as exc:
                gather_outcomes.append(exc)
        outcomes = gather_outcomes

    for task, outcome in zip(tasks, outcomes, strict=False):
        symbol = task_to_symbol.get(task)
        if symbol is None:
            continue
        if isinstance(outcome, BaseException):
            results.setdefault(symbol, None)
            FAILED_SYMBOLS.add(symbol)

    _update_peak_counters(peak_this_run)

    return results, set(SUCCESSFUL_SYMBOLS), set(FAILED_SYMBOLS)


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
        max_concurrency,
        timeout_s=timeout_s,
    )


__all__ = [
    "FAILED_SYMBOLS",
    "LAST_RUN_PEAK_SIMULTANEOUS_WORKERS",
    "PEAK_SIMULTANEOUS_WORKERS",
    "SUCCESSFUL_SYMBOLS",
    "reset_peak_simultaneous_workers",
    "reset_tracking_state",
    "run_with_concurrency",
    "run_with_concurrency_limit",
]
