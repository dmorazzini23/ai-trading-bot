"""Lightweight re-exports for utility helpers.

This module avoids importing heavy dependencies at package import time by
deferring optional components to ``__getattr__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import importlib

from .timing import HTTP_TIMEOUT, clamp_timeout, sleep  # AI-AGENT-REF: small re-exports
from .optdeps import OptionalDependencyError, module_ok  # AI-AGENT-REF: tiny helpers

_BASE_EXPORTS = {
    "EASTERN_TZ",
    "ensure_utc",
    "get_free_port",
    "get_pid_on_port",
    "health_check",
    "is_market_open",
    "market_open_between",
    "log_warning",
    "model_lock",
    "safe_to_datetime",
    "validate_ohlcv",
    "SUBPROCESS_TIMEOUT_DEFAULT",
    "safe_subprocess_run",
}

__all__ = tuple(sorted({
    "HTTP_TIMEOUT",
    "clamp_timeout",
    "sleep",
    "OptionalDependencyError",
    "module_ok",
    *(_BASE_EXPORTS),
    # lazy re-exports below
    "http",
    "retry",
    "timing",
    "device",
    "pathlib_shim",
    "datetime",
    "capital_scaling",
}))

_LAZY_MAP = {
    "http": ("ai_trading.utils.http", None),
    "retry": ("ai_trading.utils.retry", None),
    "timing": ("ai_trading.utils.timing", None),
    "device": ("ai_trading.utils.device", None),
    "pathlib_shim": ("ai_trading.utils.pathlib_shim", None),
    "datetime": ("ai_trading.utils.datetime", None),
    "capital_scaling": ("ai_trading.utils.capital_scaling", None),

    # light helpers that live in base.py; expose them lazily
    "EASTERN_TZ": ("ai_trading.utils.base", "EASTERN_TZ"),
    "ensure_utc": ("ai_trading.utils.base", "ensure_utc"),
    "get_free_port": ("ai_trading.utils.base", "get_free_port"),
    "get_pid_on_port": ("ai_trading.utils.base", "get_pid_on_port"),
    "health_check": ("ai_trading.utils.base", "health_check"),
    "is_market_open": ("ai_trading.utils.base", "is_market_open"),
    "market_open_between": ("ai_trading.utils.base", "market_open_between"),
    "log_warning": ("ai_trading.utils.base", "log_warning"),
    "model_lock": ("ai_trading.utils.base", "model_lock"),
    "safe_to_datetime": ("ai_trading.utils.base", "safe_to_datetime"),
    "validate_ohlcv": ("ai_trading.utils.base", "validate_ohlcv"),
    "SUBPROCESS_TIMEOUT_DEFAULT": ("ai_trading.utils.base", "SUBPROCESS_TIMEOUT_DEFAULT"),
    "safe_subprocess_run": ("ai_trading.utils.base", "safe_subprocess_run"),
}

if TYPE_CHECKING:  # pragma: no cover - for static analyzers only
    from . import http as http  # type: ignore
    from . import retry as retry  # type: ignore
    from . import timing as timing  # type: ignore
    from . import device as device  # type: ignore
    from . import pathlib_shim as pathlib_shim  # type: ignore
    from . import datetime as datetime  # type: ignore
    from .base import (  # type: ignore
        EASTERN_TZ,
        ensure_utc,
        get_free_port,
        get_pid_on_port,
        health_check,
        is_market_open,
        market_open_between,
        log_warning,
        model_lock,
        safe_to_datetime,
        validate_ohlcv,
        SUBPROCESS_TIMEOUT_DEFAULT,
        safe_subprocess_run,
    )


def __getattr__(name: str) -> Any:  # AI-AGENT-REF: importlib-based lazy loader
    target = _LAZY_MAP.get(name)
    if not target:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    module_name, attr = target
    mod = importlib.import_module(module_name)
    obj = getattr(mod, attr) if attr else mod
    globals()[name] = obj  # cache for subsequent access
    return obj

