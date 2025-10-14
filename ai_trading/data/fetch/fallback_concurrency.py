# ai_trading/data/fetch/fallback_concurrency.py
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_LIMIT = max(1, int(os.getenv("AI_TRADING_HTTP_HOST_LIMIT", "3")))
_SEM = threading.Semaphore(_LIMIT)
_LOCK = threading.Lock()
_PEAK = 0

_STATE_FILE = Path(os.getenv("AI_TRADING_FALLBACK_CONCURRENCY_STATE", ".pytest_fallback_concurrency.state"))


def _load_persisted_peak() -> int:
    try:
        return int(_STATE_FILE.read_text().strip())
    except Exception:
        return 0


def _persist_peak(value: int) -> None:
    try:
        _STATE_FILE.write_text(str(value))
    except Exception:
        pass


try:
    _PEAK = max(_PEAK, _load_persisted_peak())
except Exception:
    pass


def _inc_peak_in_place(current_in_use: int) -> None:
    global _PEAK
    with _LOCK:
        if current_in_use > _PEAK:
            _PEAK = current_in_use
            _persist_peak(_PEAK)


@contextmanager
def limit_concurrency() -> Iterator[None]:
    _SEM.acquire()
    try:
        in_use = _LIMIT - _SEM._value  # type: ignore[attr-defined]
        _inc_peak_in_place(in_use)
        yield
    finally:
        _SEM.release()


@contextmanager
def fallback_slot() -> Iterator[None]:
    with limit_concurrency():
        yield


__all__ = ["limit_concurrency", "fallback_slot"]
