from __future__ import annotations
from importlib import import_module

def try_import(module_name: str) -> tuple[object | None, bool, Exception | None]:
    try:
        m = import_module(module_name)
        return (m, True, None)
    except (KeyError, ValueError, TypeError) as e:
        return (None, False, e)

def get_yfinance():
    mod, ok, _ = try_import('yfinance')
    return mod if ok else None

def has_yfinance() -> bool:
    _, ok, _ = try_import('yfinance')
    return ok