from __future__ import annotations

"""Utility helpers for managing placeholder processes.

These functions are intentionally side-effect free and are primarily used by
unit tests to verify module import paths.
"""

from typing import Dict


def start_process(name: str) -> Dict[str, str]:
    """Return a started process descriptor without spawning anything."""
    return {"status": "started", "name": name}


def stop_process(name: str) -> Dict[str, str]:
    """Return a stopped process descriptor without killing anything."""
    return {"status": "stopped", "name": name}

