from __future__ import annotations
try:
    import psutil
    _HAS_PSUTIL = True
except (KeyError, ValueError, TypeError):
    psutil = None
    _HAS_PSUTIL = False


def snapshot_basic() -> dict[str, float | bool]:
    """Return a minimal health snapshot that works even without psutil."""
    data: dict[str, float | bool] = {"has_psutil": _HAS_PSUTIL}
    if _HAS_PSUTIL:
        try:
            data.update({
                "cpu_percent": psutil.cpu_percent(interval=None),
                "mem_percent": psutil.virtual_memory().percent,
            })
        except (KeyError, ValueError, TypeError):
            pass
    return data


__all__ = ["snapshot_basic"]
