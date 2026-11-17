import atexit
import threading
from concurrent.futures import ThreadPoolExecutor

from ai_trading.config.management import get_env

_EXECUTORS: dict[str, ThreadPoolExecutor] = {}
_LOCK = threading.Lock()


def _cfg_int(name: str, default: int) -> int:
    try:
        v = get_env(name, None, cast=str, resolve_aliases=False)
        return int(v) if v is not None else default
    except (KeyError, ValueError, TypeError):
        return default


def shutdown_all(wait: bool=True):
    with _LOCK:
        for ex in _EXECUTORS.values():
            ex.shutdown(wait=wait, cancel_futures=not wait)
        _EXECUTORS.clear()
atexit.register(shutdown_all)
