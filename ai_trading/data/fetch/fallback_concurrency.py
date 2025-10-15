# ai_trading/data/fetch/fallback_concurrency.py
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_DEFAULT_LIMIT = 3
_LIMIT_CONDITION = threading.Condition(threading.RLock())
_ENV_SNAPSHOT: tuple[str | None, int] | None = None
_LIMIT: int = _DEFAULT_LIMIT
_CURRENT_IN_USE = 0
_LOCK = threading.Lock()
_PEAK = 0

_STATE_FILE = Path(os.getenv("AI_TRADING_FALLBACK_CONCURRENCY_STATE", ".pytest_fallback_concurrency.state"))


def _resolve_limit() -> tuple[str | None, int]:
    raw = os.getenv("AI_TRADING_HTTP_HOST_LIMIT")
    try:
        parsed = int(str(raw).strip()) if raw is not None else None
    except (TypeError, ValueError):
        parsed = None
    if parsed is None or parsed <= 0:
        parsed = _DEFAULT_LIMIT
    return raw, max(1, parsed)


def _ensure_limit_updated() -> None:
    raw, limit = _resolve_limit()
    with _LIMIT_CONDITION:
        global _ENV_SNAPSHOT, _LIMIT
        previous = _ENV_SNAPSHOT
        if previous is not None and previous[0] == raw and previous[1] == limit:
            return
        _ENV_SNAPSHOT = (raw, limit)
        _LIMIT = limit
        _LIMIT_CONDITION.notify_all()


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
    _ensure_limit_updated()
    with _LIMIT_CONDITION:
        global _CURRENT_IN_USE
        while _CURRENT_IN_USE >= _LIMIT:
            _LIMIT_CONDITION.wait()
        _CURRENT_IN_USE += 1
        in_use = _CURRENT_IN_USE
    _inc_peak_in_place(in_use)
    try:
        yield
    finally:
        with _LIMIT_CONDITION:
            if _CURRENT_IN_USE > 0:
                _CURRENT_IN_USE -= 1
            _LIMIT_CONDITION.notify_all()


@contextmanager
def fallback_slot() -> Iterator[None]:
    with limit_concurrency():
        yield


try:
    _ensure_limit_updated()
except Exception:
    pass


__all__ = ["limit_concurrency", "fallback_slot"]
