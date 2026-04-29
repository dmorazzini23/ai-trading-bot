"""Runtime utilities."""

from __future__ import annotations

from typing import Any

__all__ = ["build_runtime", "get_max_position_size"]


def __getattr__(name: str) -> Any:
    if name == "build_runtime":
        from .params import build_runtime

        return build_runtime
    if name == "get_max_position_size":
        from .max_position_size import get_max_position_size

        return get_max_position_size
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
