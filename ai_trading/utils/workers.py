import atexit
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
_EXECUTORS: dict[str, ThreadPoolExecutor] = {}
_LOCK = threading.Lock()

def _cfg_int(name: str, default: int) -> int:
    try:
        v = os.getenv(name)
        return int(v) if v is not None else default
    except (KeyError, ValueError, TypeError):
        return default

def get_executor(name: str, max_workers: int | None=None) -> ThreadPoolExecutor:
    max_workers = max_workers or _cfg_int('WORKER_MAX_THREADS', 8)
    with _LOCK:
        ex = _EXECUTORS.get(name)
        if ex is None:
            ex = ThreadPoolExecutor(max_workers=max(1, int(max_workers)), thread_name_prefix=f'{name}-w')
            _EXECUTORS[name] = ex
    return ex

def submit_background(name: str, fn, *args, **kwargs) -> Future:
    return get_executor(name).submit(fn, *args, **kwargs)

def map_background(name: str, fn, iterable):
    ex = get_executor(name)
    return list(ex.map(fn, iterable))

def shutdown_all(wait: bool=True):
    with _LOCK:
        for ex in _EXECUTORS.values():
            ex.shutdown(wait=wait, cancel_futures=not wait)
        _EXECUTORS.clear()
atexit.register(shutdown_all)