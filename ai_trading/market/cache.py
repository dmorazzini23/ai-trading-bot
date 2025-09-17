from __future__ import annotations
from ai_trading.logging import get_logger
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = get_logger(__name__)

# pandas and its optional parquet engines are heavy and may not be present in
# minimal environments.  Import pandas if available and degrade gracefully
# otherwise.
try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - import guard
    pd = None
_lock = threading.RLock()
_mem: dict[str, tuple[float, Any]] = {}

def _now() -> float:
    return time.time()

def _key(symbol: str, tf: str, start: str, end: str) -> str:
    return f'{symbol}|{tf}|{start}|{end}'

def get_mem(symbol: str, tf: str, start: str, end: str, ttl: int) -> Any | None:
    k = _key(symbol, tf, start, end)
    with _lock:
        v = _mem.get(k)
        if not v:
            return None
        ts, obj = v
        # Use `>=` so a TTL of 0 expires immediately
        if _now() - ts >= ttl:
            _mem.pop(k, None)
            return None
        if pd is not None and hasattr(obj, 'copy'):
            return obj.copy()
        return obj

def put_mem(symbol: str, tf: str, start: str, end: str, obj: Any) -> None:
    k = _key(symbol, tf, start, end)
    with _lock:
        _mem[k] = (_now(), obj if pd is None or not hasattr(obj, 'copy') else obj.copy())

def disk_path(cache_dir: str, symbol: str, tf: str, start: str, end: str) -> Path:
    safe = f'{symbol}_{tf}_{start}_{end}'.replace(':', '-').replace('/', '-')
    return Path(cache_dir) / f'{safe}.parquet'

def get_disk(cache_dir: str, symbol: str, tf: str, start: str, end: str) -> Any | None:
    if pd is None:
        return None
    p = disk_path(cache_dir, symbol, tf, start, end)
    if p.exists():
        try:
            return pd.read_parquet(p)
        except ImportError as e:
            logger.debug('Parquet engine missing for %s: %s', p, e)
        except (
            pd.errors.EmptyDataError,
            KeyError,
            ValueError,
            TypeError,
            OSError,
            PermissionError,
        ) as e:
            logger.debug('Failed to read cache file %s: %s', p, e)
            return None
    # Fallback to CSV when parquet read is unavailable
    p_csv = p.with_suffix('.csv')
    if not p_csv.exists():
        return None
    try:
        return pd.read_csv(p_csv)
    except (
        pd.errors.EmptyDataError,
        KeyError,
        ValueError,
        TypeError,
        OSError,
        PermissionError,
    ) as e:
        logger.debug('Failed to read CSV cache file %s: %s', p_csv, e)
        return None

def put_disk(cache_dir: str, symbol: str, tf: str, start: str, end: str, df: Any) -> None:
    if pd is None:
        return
    p = disk_path(cache_dir, symbol, tf, start, end)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(p, index=False)
        return
    except ImportError as e:
        logger.debug('Parquet engine missing for %s: %s', p, e)
    except (
        pd.errors.EmptyDataError,
        KeyError,
        ValueError,
        TypeError,
        OSError,
        PermissionError,
    ) as e:
        logger.debug('Failed to write cache file %s: %s', p, e)
        return
    # Fallback to CSV when parquet write is unavailable
    p_csv = p.with_suffix('.csv')
    try:
        df.to_csv(p_csv, index=False)
    except (
        pd.errors.EmptyDataError,
        KeyError,
        ValueError,
        TypeError,
        OSError,
        PermissionError,
    ) as e:
        logger.debug('Failed to write CSV cache file %s: %s', p_csv, e)


def _normalize_key(
    key: str | tuple[str, str, str, str],
) -> tuple[str, tuple[str, str, str, str], tuple[str, str, str, str] | None]:
    """Return a string cache key alongside normalized cache tuple components."""

    if isinstance(key, tuple):
        symbol, tf, start, end = key
        cache_key = _key(symbol, tf, start, end)
        return cache_key, (symbol, tf, start, end), (symbol, tf, start, end)

    mem_symbol = f"__key__:{key}"
    placeholder = (mem_symbol, "__", "__", "__")
    return key, placeholder, None


def get_or_load(
    *,
    key: str | tuple[str, str, str, str],
    loader: Callable[[], Any],
    ttl: int,
    cache_dir: str | Path | None = None,
    disk_enabled: bool | None = None,
    disk_key: tuple[str, str, str, str] | None = None,
    mem_enabled: bool = True,
) -> Any:
    """Retrieve cached data or load it, updating both memory and disk stores."""

    _cache_key, mem_parts, default_disk_parts = _normalize_key(key)

    disk_parts = disk_key if disk_key is not None else default_disk_parts

    should_use_disk = bool(cache_dir) if disk_enabled is None else bool(disk_enabled and cache_dir)

    cached: Any | None = None
    if mem_enabled:
        cached = get_mem(*mem_parts, ttl=ttl)
        if cached is not None:
            return cached

    if should_use_disk and disk_parts is not None:
        cached = get_disk(str(cache_dir), *disk_parts)
        if cached is not None:
            if mem_enabled:
                put_mem(*mem_parts, cached)
            return cached

    result = loader()

    if mem_enabled:
        put_mem(*mem_parts, result)

    if should_use_disk and disk_parts is not None and result is not None:
        if hasattr(result, "to_parquet") or hasattr(result, "to_csv"):
            put_disk(str(cache_dir), *disk_parts, result)

    return result
