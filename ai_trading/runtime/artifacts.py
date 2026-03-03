"""Helpers for resolving runtime artifact paths consistently."""

from __future__ import annotations

from pathlib import Path

from ai_trading.config.management import get_env


_REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_runtime_artifact_path(
    path_value: str | Path | None,
    *,
    default_relative: str,
) -> Path:
    """Resolve ``path_value`` against runtime roots when relative.

    Resolution order for relative paths:
    1. ``AI_TRADING_DATA_DIR`` (preferred runtime data root)
    2. ``STATE_DIRECTORY`` (systemd state dir compatibility)
    3. repository root
    """

    raw_value = str(path_value or "").strip() or str(default_relative)
    target = Path(raw_value).expanduser()
    if target.is_absolute():
        return target.resolve()

    for env_key in ("AI_TRADING_DATA_DIR", "STATE_DIRECTORY"):
        root_raw = str(get_env(env_key, "", cast=str) or "").strip()
        if not root_raw:
            continue
        root = Path(root_raw.split(":")[0]).expanduser()
        if root.is_absolute():
            return (root / target).resolve()

    return (_REPO_ROOT / target).resolve()


__all__ = ["resolve_runtime_artifact_path"]
