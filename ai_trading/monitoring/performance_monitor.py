from __future__ import annotations
from dataclasses import dataclass

# AI-AGENT-REF: stub performance monitor for tests
@dataclass
class ResourceMonitor:
    """Lightweight stub used by tests."""

    def snapshot(self) -> dict:  # pragma: no cover
        return {"cpu": 0.0, "mem": 0.0}


def get_performance_monitor() -> ResourceMonitor:  # pragma: no cover - test shim
    """Return a stub monitor instance."""
    return ResourceMonitor()


def start_performance_monitoring() -> None:  # pragma: no cover - test shim
    """No-op monitoring starter."""
    return None
