from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

import time as _time
from time import perf_counter as _perf
from typing import Optional, Union

from .sleep import _real_sleep, sleep


def _managed_env(name: str, default: str) -> str:
    """Resolve env keys via config management without import-time hard coupling."""

    try:
        from ai_trading.config.management import get_env as _get_env

        value = _get_env(name, default, cast=str, resolve_aliases=False)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        value = default
    if value in (None, ""):
        return default
    return str(value)


def _parse_timeout(value: str, default: float = 10.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


# Prefer HTTP_TIMEOUT when present; fallback to AI_HTTP_TIMEOUT env.
HTTP_TIMEOUT: Union[int, float] = _parse_timeout(
    _managed_env("HTTP_TIMEOUT", "") or _managed_env("AI_HTTP_TIMEOUT", "") or "10"
)  # AI-AGENT-REF: canonical timeout across runtime


def clamp_timeout(value: Optional[float]) -> float:
    """Return a sane timeout, falling back to HTTP_TIMEOUT when None/invalid."""  # AI-AGENT-REF: clamp helper
    try:
        if value is None:
            return HTTP_TIMEOUT
        v = float(value)
        return v if v > 0 else HTTP_TIMEOUT
    except (TypeError, ValueError):
        return HTTP_TIMEOUT


def _robust_sleep(seconds: Union[int, float]) -> float:
    """Block for at least ~10ms even under monkeypatched time.sleep.

    Uses the original OS sleep captured at import time and a short
    perf_counter-based busy wait to ensure measurable elapsed time.
    """  # AI-AGENT-REF: deterministic sleep

    try:
        s = float(seconds)
    except (TypeError, ValueError):
        s = 0.0
    target = max(s, 0.01)
    start = _perf()
    _real_sleep(target)
    # Ensure we cross ~10ms even if scheduler wakes early; cap iterations to avoid hangs
    _tries = 0
    while (_perf() - start) < 0.01 and _tries < 5:
        _real_sleep(0.005)
        _tries += 1
    return _perf() - start


def _system_sleep(seconds: Union[int, float]) -> float:
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        s = 0.0
    if s <= 0:
        return 0.0
    start = _perf()
    _time.sleep(s)
    return _perf() - start


_force_local_sleep = str(
    _managed_env("AI_TRADING_FORCE_LOCAL_SLEEP", "1")
).lower() in {"1", "true", "yes", "on"}
if _force_local_sleep:
    sleep = _robust_sleep  # type: ignore[assignment]
else:  # pragma: no cover
    sleep = _system_sleep


__all__ = ["HTTP_TIMEOUT", "clamp_timeout", "sleep"]
