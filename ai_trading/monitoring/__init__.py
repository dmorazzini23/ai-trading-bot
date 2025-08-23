from __future__ import annotations

# Thin monitoring facade exposing only the canonical health-check API.
from .system_health import snapshot_basic

__all__ = ["snapshot_basic"]
