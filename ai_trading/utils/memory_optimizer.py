"""Low-overhead runtime memory telemetry helpers."""

from __future__ import annotations

import gc
import importlib
import json
import os
import resource
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_trading.config.management import get_env
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path

_DEFAULT_SAMPLE_PATH = "runtime/memory_samples.jsonl"
_DEFAULT_MAX_BYTES = 5_000_000


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _get_bool_env(name: str, default: bool) -> bool:
    try:
        value = get_env(name, None, cast=str, resolve_aliases=False)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        return default
    if value is None:
        return default
    return _truthy(value)


def _get_int_env(name: str, default: int) -> int:
    try:
        raw = get_env(name, None, cast=str, resolve_aliases=False)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        return default
    try:
        return int(str(raw).strip()) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _get_float_env(name: str, default: float | None) -> float | None:
    try:
        raw = get_env(name, None, cast=str, resolve_aliases=False)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        return default
    try:
        return float(str(raw).strip()) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _mb(value_bytes: int | float | None) -> float | None:
    if value_bytes is None:
        return None
    return round(float(value_bytes) / (1024.0 * 1024.0), 3)


def _resource_maxrss_mb() -> float | None:
    try:
        maxrss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        return None
    if maxrss <= 0:
        return None
    # Linux reports ru_maxrss in KiB; macOS reports bytes.
    if maxrss > 10_000_000:
        return round(maxrss / (1024.0 * 1024.0), 3)
    return round(maxrss / 1024.0, 3)


def _read_proc_meminfo() -> dict[str, float | None]:
    values: dict[str, float | None] = {
        "mem_total_mb": None,
        "mem_available_mb": None,
        "swap_total_mb": None,
        "swap_free_mb": None,
    }
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return values
    mapping = {
        "MemTotal": "mem_total_mb",
        "MemAvailable": "mem_available_mb",
        "SwapTotal": "swap_total_mb",
        "SwapFree": "swap_free_mb",
    }
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        key = mapping.get(name)
        if key is None:
            continue
        parts = rest.strip().split()
        try:
            values[key] = round(float(parts[0]) / 1024.0, 3)
        except (IndexError, TypeError, ValueError):
            values[key] = None
    return values


def _process_memory_from_psutil() -> dict[str, float | int | None]:
    try:
        psutil = importlib.import_module("psutil")
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        return {
            "rss_mb": None,
            "vms_mb": None,
            "shared_mb": None,
            "threads": threading.active_count(),
            "open_fds": _open_fd_count(),
        }
    return {
        "rss_mb": _mb(getattr(memory_info, "rss", None)),
        "vms_mb": _mb(getattr(memory_info, "vms", None)),
        "shared_mb": _mb(getattr(memory_info, "shared", None)),
        "threads": int(process.num_threads()),
        "open_fds": int(process.num_fds()) if hasattr(process, "num_fds") else _open_fd_count(),
    }


def _open_fd_count() -> int | None:
    try:
        return len(list(Path("/proc/self/fd").iterdir()))
    except OSError:
        return None


def _resolve_sample_path(sample_path: str | Path | None) -> Path:
    configured = sample_path
    if configured is None:
        try:
            configured = get_env(
                "AI_TRADING_MEMORY_SAMPLE_PATH",
                _DEFAULT_SAMPLE_PATH,
                cast=str,
            )
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            configured = _DEFAULT_SAMPLE_PATH
    return resolve_runtime_artifact_path(
        configured,
        default_relative=_DEFAULT_SAMPLE_PATH,
        for_write=True,
    )


def _compact_jsonl(path: Path, max_bytes: int) -> None:
    if max_bytes <= 0:
        return
    try:
        size = path.stat().st_size
    except OSError:
        return
    if size <= max_bytes:
        return
    keep_bytes = max(1024, int(max_bytes * 0.75))
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, size - keep_bytes))
            data = handle.read(keep_bytes)
    except OSError:
        return
    if data.startswith(b"\n"):
        trimmed = data[1:]
    else:
        _, sep, remainder = data.partition(b"\n")
        trimmed = remainder if sep else data
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(trimmed)
            if trimmed and not trimmed.endswith(b"\n"):
                tmp.write(b"\n")
        os.replace(tmp_name, path)
    except OSError:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass


def _append_sample(sample: dict[str, Any], sample_path: str | Path | None, max_bytes: int) -> Path | None:
    try:
        path = _resolve_sample_path(sample_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _compact_jsonl(path, max_bytes)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
        _compact_jsonl(path, max_bytes)
        return path
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        return None


def _level_for_rss(rss_mb: float | None) -> str:
    if rss_mb is None:
        return "unknown"
    warning = _get_float_env("AI_TRADING_MEMORY_WARN_MB", 1_200.0)
    critical = _get_float_env("AI_TRADING_MEMORY_CRITICAL_MB", 1_600.0)
    if critical is not None and rss_mb >= critical:
        return "critical"
    if warning is not None and rss_mb >= warning:
        return "warning"
    return "normal"


def enable_low_memory_mode() -> None:
    """Run a conservative cleanup pass for callers that opt into it."""

    gc.collect()


def report_memory_use(
    *,
    cycle_index: int | None = None,
    closed: bool | None = None,
    interval_s: float | None = None,
    write_sample: bool = False,
    sample_path: str | Path | None = None,
    max_bytes: int | None = None,
    collect: bool = False,
) -> dict[str, Any]:
    """Return a stable memory snapshot and optionally append it to JSONL."""

    objects_collected = gc.collect() if collect else 0
    process = _process_memory_from_psutil()
    sample: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "pid": os.getpid(),
        "rss_mb": process.get("rss_mb"),
        "vms_mb": process.get("vms_mb"),
        "shared_mb": process.get("shared_mb"),
        "maxrss_mb": _resource_maxrss_mb(),
        "threads": process.get("threads"),
        "open_fds": process.get("open_fds"),
        "gc_objects": len(gc.get_objects()),
        "gc_counts": list(gc.get_count()),
        "objects_collected": objects_collected,
        "cycle_index": cycle_index,
        "closed": closed,
        "interval_s": interval_s,
    }
    sample.update(_read_proc_meminfo())
    sample["level"] = _level_for_rss(
        sample.get("rss_mb") if isinstance(sample.get("rss_mb"), float) else None
    )
    if write_sample and _get_bool_env("AI_TRADING_MEMORY_TELEMETRY_ENABLED", True):
        resolved_max_bytes = (
            max_bytes
            if max_bytes is not None
            else _get_int_env("AI_TRADING_MEMORY_SAMPLE_MAX_BYTES", _DEFAULT_MAX_BYTES)
        )
        path = _append_sample(sample, sample_path, resolved_max_bytes)
        sample["sample_path"] = str(path) if path is not None else None
    return sample
