from __future__ import annotations

import importlib
import inspect
import threading
import time as _time
from typing import Callable

from ai_trading.utils.time import monotonic_time

__all__ = ["sleep"]

# Capture the original OS-level sleep to avoid monkeypatch interference
_real_sleep = _time.sleep

# Freezegun can replace ``time.sleep`` with a fast-forward stub. When that
# happens we want to defer the lookup to runtime so we can locate a genuine
# blocking sleep implementation as soon as it becomes available again.
_initial_freezegun_sleep = "freezegun" in (getattr(_real_sleep, "__module__", "") or "").lower()

# Hold on to the first known-good OS sleep so subsequent monkeypatches are
# ignored (the historical behaviour of this helper).
_os_level_sleep = None if _initial_freezegun_sleep else _real_sleep

try:  # pragma: no cover - optional dependency
    import freezegun.api as _freezegun_api  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _freezegun_api = None  # type: ignore


def _is_reliable_sleep(candidate: Callable | None) -> bool:
    """Return True when *candidate* is a real blocking builtin sleep."""

    if not callable(candidate):
        return False
    module_name = (getattr(candidate, "__module__", "") or "").lower()
    return inspect.isbuiltin(candidate) and "freezegun" not in module_name


def _event_wait_sleep(seconds: float) -> None:
    """Sleep via threading wait to bypass monkeypatched ``time.sleep``."""

    if seconds <= 0:
        return
    waiter = threading.Event()
    deadline = _monotonic_now() + float(seconds)
    while True:
        remaining = deadline - _monotonic_now()
        if remaining <= 0:
            return
        waiter.wait(remaining)


def _freezegun_active(runtime_sleep: Callable | None = None) -> bool:
    """Return ``True`` when Freezegun currently controls the clock."""

    if runtime_sleep is not None:
        module_name = (getattr(runtime_sleep, "__module__", "") or "").lower()
        if "freezegun" in module_name:
            return True

    if _initial_freezegun_sleep:
        return True

    if _freezegun_api is None:  # Freezegun not installed or import failed
        return False

    try:
        factories = getattr(_freezegun_api, "freeze_factories", ())
    except Exception:  # pragma: no cover - defensive
        return False
    return bool(factories)


def _monotonic_now() -> float:
    """Return a monotonic timestamp resilient to Freezegun freezes."""

    if _freezegun_api is not None and _freezegun_active():
        for attr in ("real_monotonic", "real_perf_counter", "real_time"):
            func = getattr(_freezegun_api, attr, None)
            if callable(func):
                try:
                    return float(func())
                except Exception:  # pragma: no cover - defensive
                    continue
    return monotonic_time()


def _resolve_sleep():
    """Return the most reliable sleep callable available."""

    global _os_level_sleep

    runtime_sleep = getattr(_time, "sleep", None)
    if callable(runtime_sleep):
        runtime_module = (getattr(runtime_sleep, "__module__", "") or "").lower()
        runtime_is_builtin = _is_reliable_sleep(runtime_sleep)
        if _freezegun_active(runtime_sleep):
            if runtime_is_builtin and "freezegun" not in runtime_module:
                if _os_level_sleep is None:
                    _os_level_sleep = runtime_sleep
                return runtime_sleep

            # ``time.sleep`` might be provided by Freezegun or another stub.
            # Try to reload the module to obtain a genuine blocking sleep
            # implementation.
            try:
                reloaded_time = importlib.import_module("time")
            except Exception:  # pragma: no cover - defensive guard
                reloaded_time = _time
            reload_sleep = getattr(reloaded_time, "sleep", None)
            if callable(reload_sleep):
                reload_module = (getattr(reload_sleep, "__module__", "") or "").lower()
                if _is_reliable_sleep(reload_sleep) and "freezegun" not in reload_module:
                    if _os_level_sleep is None:
                        _os_level_sleep = reload_sleep
                    return reload_sleep
        elif _os_level_sleep is None and runtime_is_builtin and "freezegun" not in runtime_module:
            _os_level_sleep = runtime_sleep

    if _is_reliable_sleep(_os_level_sleep):
        return _os_level_sleep
    if _is_reliable_sleep(_real_sleep):
        return _real_sleep
    return _event_wait_sleep


def sleep(seconds: float | int) -> float:
    """Sleep for ``seconds`` using :func:`time.sleep` and report elapsed time.

    ``time.sleep`` is captured at import time so monkeypatching ``time.sleep``
    later will not affect this helper. The elapsed duration is measured using
    a monotonic clock resilient to Freezegun freezes and returned to the
    caller. If ``seconds`` is zero or negative, the function returns ``0.0``
    without sleeping.
    """
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return 0.0
    if s <= 0:
        return 0.0
    start = _monotonic_now()
    sleeper = _resolve_sleep()
    sleeper(s)
    return _monotonic_now() - start
