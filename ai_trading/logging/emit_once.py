"""Helpers to emit a log record only once per UTC day per process."""
from __future__ import annotations

import logging
import threading
from datetime import UTC, date, datetime
from logging import Logger, LoggerAdapter
from typing import Any, overload

_emitted: dict[str, date] = {}
_lock = threading.Lock()


def _utc_today() -> date:
    return datetime.now(UTC).date()


def _should_emit(key: str) -> bool:
    today = _utc_today()
    with _lock:
        last = _emitted.get(key)
        if last == today:
            return False
        _emitted[key] = today
        return True


@overload
def emit_once(key: str, /) -> bool: ...


@overload
def emit_once(
    logger: Logger | LoggerAdapter,
    key: str,
    level: str,
    msg: str,
    /,
    **extra: Any,
) -> bool: ...


def _coerce_level(level: str | int) -> int:
    """Return the numeric logging level for ``level``.

    ``logging.getLevelName`` returns a string when it cannot resolve the value,
    so we normalise and validate the input before returning the numeric level.
    """

    if isinstance(level, int):
        return level
    if isinstance(level, str):
        resolved = logging.getLevelName(level.strip().upper())
        if isinstance(resolved, int):
            return resolved
    raise ValueError(f"Unsupported log level: {level!r}")


def emit_once(*args: Any, **extra: Any) -> bool:
    """Emit ``msg`` at ``level`` once per UTC day keyed by ``key``.

    The helper supports two call modes:

    * ``emit_once("KEY")`` returns ``True`` the first time ``KEY`` is seen on a
      given UTC day and ``False`` for subsequent calls.
    * ``emit_once(logger, "KEY", "info", "message")`` logs ``message`` using
      ``logger`` on the first call per UTC day and returns ``True`` when the
      message was emitted.
    """

    if not args:
        raise TypeError("emit_once expects at least one positional argument")

    first = args[0]
    if isinstance(first, (Logger, LoggerAdapter)):
        if len(args) != 4:
            raise TypeError(
                "emit_once(logger, key, level, msg) requires four positional arguments"
            )
        logger, key, level, msg = first, str(args[1]), args[2], str(args[3])
        if not _should_emit(key):
            return False
        numeric_level = _coerce_level(level)
        if extra:
            logger.log(numeric_level, str(msg), extra=extra)
        else:
            logger.log(numeric_level, str(msg))
        return True

    if len(args) != 1:
        raise TypeError("emit_once(key) expects exactly one positional argument when no logger is provided")
    if extra:
        raise TypeError("emit_once(key) does not accept keyword arguments")
    key = str(first)
    return _should_emit(key)



def reset_emit_once_state() -> None:
    """Clear the per-process emit-once tracking cache.

    This is intended for use in tests that need to ensure a clean slate between
    runs without relying on internal module state.
    """

    with _lock:
        _emitted.clear()


__all__ = ["emit_once", "reset_emit_once_state"]
