"""Host-level concurrency limit helpers with peak tracking."""

from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

import asyncio
import json
from pathlib import Path
import threading
from typing import cast

from ai_trading.config.management import get_env
from ai_trading.logging import get_logger
from ai_trading.http import pooling

logger = get_logger(__name__)

_DEFAULT_HOST_KEY = getattr(pooling, "_DEFAULT_HOST_KEY", "__default__")
_FALLBACK_LOCK = threading.Lock()
_FALLBACK_SYNC_LIMITERS: dict[str, threading.Semaphore] = {}
_FALLBACK_ASYNC_LIMITERS: dict[str, asyncio.Semaphore] = {}

_PEAK_PATH_ENV = "AI_TRADING_FALLBACK_PEAK_PATH"
_DEFAULT_PEAK_PATH = "/tmp/ai_trading_fallback_peak.json"

_COUNTER_LOCK = threading.Lock()
_IN_FLIGHT = 0
_PEAK = 0


def _peak_path() -> Path:
    raw = get_env(_PEAK_PATH_ENV, None, cast=str, resolve_aliases=False)
    if raw:
        try:
            return Path(raw)
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            logger.debug("PEAK_PATH_ENV_PARSE_FAILED", extra={"raw": raw}, exc_info=True)
    return Path(_DEFAULT_PEAK_PATH)


def _load_peak_from_disk() -> int:
    path = _peak_path()
    try:
        data = json.loads(path.read_text())
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        logger.debug("PEAK_FILE_READ_FAILED", extra={"path": str(path)}, exc_info=True)
        return 0
    value = data.get("peak") if isinstance(data, dict) else None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


with _COUNTER_LOCK:
    _PEAK = _load_peak_from_disk()


def _normalize_host(host: str | None) -> str:
    normalizer = getattr(pooling, "_normalize_host", None)
    if callable(normalizer):
        try:
            return cast(str, normalizer(host))
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            logger.debug("HOST_NORMALIZER_DELEGATE_FAILED", extra={"host": host}, exc_info=True)
    normalized = (host or "").strip().lower()
    return normalized or _DEFAULT_HOST_KEY


def current_peak() -> int:
    """Return the highest observed concurrent host operations."""

    with _COUNTER_LOCK:
        return _PEAK


__all__ = [
    "current_peak",
]
