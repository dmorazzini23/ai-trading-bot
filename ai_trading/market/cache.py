from __future__ import annotations
import time
import threading
import logging
from typing import Any, Dict, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # tests can mock

_lock = threading.RLock()
_mem: Dict[str, Tuple[float, Any]] = {}

def _now() -> float: return time.time()

def _key(symbol: str, tf: str, start: str, end: str) -> str:
    return f"{symbol}|{tf}|{start}|{end}"

def get_mem(symbol: str, tf: str, start: str, end: str, ttl: int) -> Optional[Any]:
    k = _key(symbol, tf, start, end)
    with _lock:
        v = _mem.get(k)
        if not v: return None
        ts, obj = v
        if _now() - ts > ttl:
            _mem.pop(k, None)
            return None
        # return a defensive copy for DataFrame
        if pd is not None and hasattr(obj, "copy"): return obj.copy()
        return obj

def put_mem(symbol: str, tf: str, start: str, end: str, obj: Any) -> None:
    k = _key(symbol, tf, start, end)
    with _lock:
        _mem[k] = (_now(), obj if pd is None or not hasattr(obj, "copy") else obj.copy())

def disk_path(cache_dir: str, symbol: str, tf: str, start: str, end: str) -> Path:
    safe = f"{symbol}_{tf}_{start}_{end}".replace(":", "-").replace("/", "-")
    return Path(cache_dir) / f"{safe}.parquet"

def get_disk(cache_dir: str, symbol: str, tf: str, start: str, end: str) -> Optional[Any]:
    if pd is None: return None
    p = disk_path(cache_dir, symbol, tf, start, end)
    if not p.exists(): return None
    try:
        return pd.read_parquet(p)
    except Exception as e:
        logger.debug("Failed to read cache file %s: %s", p, e)
        return None

def put_disk(cache_dir: str, symbol: str, tf: str, start: str, end: str, df: Any) -> None:
    if pd is None: return
    p = disk_path(cache_dir, symbol, tf, start, end)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(p, index=False)
    except Exception as e:
        logger.debug("Failed to write cache file %s: %s", p, e)