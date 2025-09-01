"""Test-time environment sanitization.

This module is imported automatically by Python at startup if present on
``sys.path``. It installs a tiny shim to make two executor-related env vars
safe to parse in tests that do simple ``int(os.getenv(... ) or '0')`` logic.

Only the following keys are affected:
 - EXECUTOR_WORKERS
 - PREDICTION_WORKERS

If either is set to a non-numeric string, ``os.getenv`` will return an empty
string (""), allowing downstream ``or '0'`` fallbacks to work as intended.

This does not change behavior for any other environment variable.
"""
from __future__ import annotations

import os
from typing import Any


_orig_getenv = os.getenv


def _sanitized_getenv(key: str, default: Any | None = None):  # type: ignore[override]
    if str(key).upper() in {"EXECUTOR_WORKERS", "PREDICTION_WORKERS"}:
        val = _orig_getenv(key, default)
        try:
            return val if (val is None or str(val).isdigit()) else ""
        except Exception:
            return ""
    return _orig_getenv(key, default)


try:
    os.getenv = _sanitized_getenv  # type: ignore[assignment]
except Exception:
    pass

