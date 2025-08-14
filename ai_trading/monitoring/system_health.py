from __future__ import annotations

# AI-AGENT-REF: optional psutil snapshot
try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:
    psutil = None  # type: ignore
    _HAS_PSUTIL = False


def snapshot_basic() -> dict[str, float | bool]:
    """Return a minimal health snapshot that works even without psutil."""
    data: dict[str, float | bool] = {"has_psutil": _HAS_PSUTIL}
    if _HAS_PSUTIL:
        try:
            data.update(
                {
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "mem_percent": psutil.virtual_memory().percent,
                }
            )
        except Exception:
            # keep minimal snapshot if psutil misbehaves
            pass
    return data
