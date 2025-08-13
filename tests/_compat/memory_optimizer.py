"""
Test-only shim for missing `memory_optimizer` dependency.
Provides no-op helpers so CI/test collection succeeds without
affecting production behavior. Remove when real impl is present.
"""

from __future__ import annotations

from typing import Any

# AI-AGENT-REF: test-only memory optimizer shim

__all__ = [
    "is_available",
    "optimize_df",
    "optimize_array",
    "gc_collect",
    "MemoryOptimizer",
    "memory_profile",
    "optimize_memory",
    "emergency_memory_cleanup",
]


def is_available() -> bool:
    """Always False in tests; signals no real optimizer is present."""
    return False


def optimize_df(df: Any, *_, **__) -> Any:
    """Return the input unchanged (deterministic no-op)."""
    return df


def optimize_array(arr: Any, *_, **__) -> Any:
    """Return the input unchanged (deterministic no-op)."""
    return arr


def gc_collect() -> None:
    """No-op hook for symmetry with real implementations."""
    return None


class MemoryOptimizer:
    """No-op context/utility mirroring a likely prod API."""

    enabled: bool = False

    def __init__(self, *_, **__):
        self.enabled = False

    def enable(self) -> None:
        self.enabled = True  # harmless flag for tests

    def disable(self) -> None:
        self.enabled = False

    # Convenience aliases commonly used in code
    def optimize_df(self, df: Any, *_, **__) -> Any:
        return df

    def optimize_array(self, arr: Any, *_, **__) -> Any:
        return arr


def memory_profile(func: Any) -> Any:
    """Decorator passthrough used in tests."""
    return func


def optimize_memory() -> dict[str, Any]:
    """Return empty stats to mirror prod API."""
    return {}


def emergency_memory_cleanup() -> dict[str, Any]:
    """Return empty stats to mirror prod API."""
    return {}

