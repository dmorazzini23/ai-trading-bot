from __future__ import annotations

from importlib import import_module
from typing import Optional, Tuple

# AI-AGENT-REF: helper for lazy optional imports

def try_import(module_name: str) -> Tuple[Optional[object], bool, Optional[Exception]]:
    try:
        m = import_module(module_name)
        return m, True, None
    except Exception as e:  # intentionally broad to capture env issues
        return None, False, e


def get_yfinance():
    mod, ok, _ = try_import("yfinance")
    return mod if ok else None


def has_yfinance() -> bool:
    _, ok, _ = try_import("yfinance")
    return ok
