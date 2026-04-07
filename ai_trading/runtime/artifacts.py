"""Helpers for resolving runtime artifact paths consistently."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ai_trading.config.management import get_env


_REPO_ROOT = Path(__file__).resolve().parents[2]
_APP_NAME = "ai-trading-bot"


def _iter_runtime_roots() -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()
    for env_key in ("AI_TRADING_DATA_DIR", "STATE_DIRECTORY"):
        root_raw = str(get_env(env_key, "", cast=str) or "").strip()
        if not root_raw:
            continue
        root = Path(root_raw.split(":")[0]).expanduser()
        if not root.is_absolute():
            continue
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        roots.append(root)

    try:
        from ai_trading.paths import DATA_DIR

        managed_root = Path(DATA_DIR).expanduser()
    except Exception:
        managed_root = None
    if managed_root is not None and managed_root.is_absolute():
        key = str(managed_root)
        if key not in seen:
            seen.add(key)
            roots.append(managed_root)
    return roots


def _fallback_runtime_root() -> Path:
    base_dir = Path(tempfile.gettempdir()).expanduser()
    return (base_dir / _APP_NAME).resolve()


def _target_parent_writable(path: Path) -> bool:
    parent = path.parent
    if parent.exists():
        return os.access(parent, os.W_OK)

    probe_parent = parent
    while not probe_parent.exists():
        if probe_parent == probe_parent.parent:
            break
        probe_parent = probe_parent.parent
    return probe_parent.exists() and os.access(probe_parent, os.W_OK)


def _resolve_relative_candidate(target: Path, root: Path) -> Path:
    return (root / target).resolve()


def resolve_runtime_artifact_path(
    path_value: str | Path | None,
    *,
    default_relative: str,
    for_write: bool = False,
) -> Path:
    """Resolve ``path_value`` against runtime roots when relative.

    Resolution order for relative paths:
    1. ``AI_TRADING_DATA_DIR`` (preferred runtime data root)
    2. ``STATE_DIRECTORY`` (systemd state dir compatibility)
    3. managed runtime data dir (``ai_trading.paths.DATA_DIR``)
    4. repository root
    """

    raw_value = str(path_value or "").strip() or str(default_relative)
    target = Path(raw_value).expanduser()
    if target.is_absolute():
        return target.resolve()

    candidates = [
        _resolve_relative_candidate(target, root)
        for root in _iter_runtime_roots()
    ]
    candidates.append((_REPO_ROOT / target).resolve())

    if not for_write:
        return candidates[0]

    for candidate in candidates:
        if _target_parent_writable(candidate):
            return candidate

    fallback = _resolve_relative_candidate(target, _fallback_runtime_root())
    fallback.parent.mkdir(parents=True, exist_ok=True)
    return fallback


__all__ = ["resolve_runtime_artifact_path"]
