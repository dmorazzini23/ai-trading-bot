from __future__ import annotations
import importlib.util as _ils
from types import ModuleType

def module_ok(name: str) -> bool:
    """Return True if *name* can be imported."""
    try:
        return _ils.find_spec(name) is not None
    except (KeyError, ValueError, TypeError, ModuleNotFoundError):
        return False

def try_import(name: str) -> ModuleType | None:
    """Attempt to import *name*, returning module or None."""
    try:
        return __import__(name)
    except (KeyError, ValueError, TypeError, ModuleNotFoundError):
        return None
