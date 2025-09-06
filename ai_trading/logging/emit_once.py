"""Helper to emit a log record only once per day per process."""
from __future__ import annotations
import threading
from datetime import date
from logging import Logger

_emitted: dict[str, tuple[date, int]] = {}
_lock = threading.Lock()

def emit_once(logger: Logger, key: str, level: str, msg: str, **extra) -> bool:
    """Emit ``msg`` at ``level`` only once per day keyed by ``key``."""
    token = f"{logger.name}:{key}"
    today = date.today()
    with _lock:
        last_date, count = _emitted.get(token, (None, 0))
        if last_date != today:
            count = 0
        count += 1
        _emitted[token] = (today, count)
        if count > 1:
            return False
    fn = getattr(logger, level.lower(), logger.info)
    fn(msg, extra=extra or None)
    return True

__all__ = ["emit_once"]
