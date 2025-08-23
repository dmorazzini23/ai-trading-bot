"""Minimal facade needed by tests; not a feature expansion."""
from __future__ import annotations
from dataclasses import dataclass
import psutil

# AI-AGENT-REF: restore test-facing performance monitor

@dataclass
class Snapshot:
    cpu: float
    rss_mb: float


class ResourceMonitor:
    def snapshot_basic(self) -> Snapshot:
        proc = psutil.Process()
        rss_mb = proc.memory_info().rss / (1024 * 1024)
        return Snapshot(cpu=psutil.cpu_percent(interval=None), rss_mb=rss_mb)

