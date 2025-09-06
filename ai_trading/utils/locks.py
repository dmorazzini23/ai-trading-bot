"""Synchronization primitives used across the project."""

from __future__ import annotations

import threading

__all__ = ["portfolio_lock"]

# Single lock instance guarding shared portfolio state.
portfolio_lock = threading.Lock()
