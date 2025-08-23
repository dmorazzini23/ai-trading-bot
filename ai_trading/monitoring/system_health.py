from __future__ import annotations
try:
    import psutil
    _HAS_PSUTIL = True
except (KeyError, ValueError, TypeError):
    psutil = None
    _HAS_PSUTIL = False

def snapshot_basic() -> dict[str, float | bool]:
    """Return a minimal health snapshot that works even without psutil."""
    data: dict[str, float | bool] = {'has_psutil': _HAS_PSUTIL}
    if _HAS_PSUTIL:
        try:
            data.update({'cpu_percent': psutil.cpu_percent(interval=None), 'mem_percent': psutil.virtual_memory().percent})
        except (KeyError, ValueError, TypeError):
            pass
    return data


class ResourceMonitor:
    """Minimal resource monitor used in tests."""  # AI-AGENT-REF: stub for public API

    def __init__(self, monitoring_interval: int=30):
        self.monitoring_interval = monitoring_interval

    def _count_trading_bot_processes(self) -> int:
        """Return a sentinel process count."""
        return 1


__all__ = ["snapshot_basic", "ResourceMonitor"]