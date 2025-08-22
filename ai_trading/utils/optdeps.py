from __future__ import annotations

# AI-AGENT-REF: helpers for optional dependencies
import importlib.util as _ils
from types import ModuleType


def module_ok(name: str) -> bool:
    """Return True if *name* can be imported."""  # AI-AGENT-REF: import guard
    try:
        return _ils.find_spec(name) is not None
    # noqa: BLE001 TODO: narrow exception
    except Exception:
        return False


def try_import(name: str) -> ModuleType | None:
    """Attempt to import *name*, returning module or None."""  # AI-AGENT-REF
    try:
        return __import__(name)
    # noqa: BLE001 TODO: narrow exception
    except Exception:
        return None
