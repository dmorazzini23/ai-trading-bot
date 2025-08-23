"""Helper to emit a log record only once per process."""
from __future__ import annotations
import threading
from logging import Logger
_emitted: set[str] = set()
_lock = threading.Lock()

def emit_once(logger: Logger, key: str, level: str, msg: str, **extra) -> bool:
    """Emit ``msg`` at ``level`` only once per process keyed by ``key``."""
    token = f'{logger.name}:{key}'
    with _lock:
        if token in _emitted:
            return False
        _emitted.add(token)
    fn = getattr(logger, level.lower(), logger.info)
    fn(msg, extra=extra or None)
    return True
__all__ = ['emit_once']