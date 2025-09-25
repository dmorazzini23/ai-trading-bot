"""Helpers to emit a log record only once per UTC day per process."""
from __future__ import annotations

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
        logger, key, level, msg = first, str(args[1]), str(args[2]), str(args[3])
        if not _should_emit(key):
            return False
        log_fn = getattr(logger, level.lower(), logger.info)
        if extra:
            log_fn(str(msg), extra=extra)
        else:
            log_fn(str(msg))
        return True

    if len(args) != 1:
        raise TypeError("emit_once(key) expects exactly one positional argument when no logger is provided")
    if extra:
        raise TypeError("emit_once(key) does not accept keyword arguments")
    key = str(first)
    return _should_emit(key)


__all__ = ["emit_once"]
