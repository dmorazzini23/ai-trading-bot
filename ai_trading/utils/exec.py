"""Execution helpers for isolated subprocess and worker environments."""
from __future__ import annotations

import os
from typing import Iterable, Mapping, Sequence

# Default list of environment variables to preserve when sanitizing.
_DEFAULT_WHITELIST = {
    "PATH",
    "PYTHONPATH",
    "HOME",
    "LANG",
    "AI_TRADING_OFFLINE_TESTS",
}


def _sanitize_executor_env(
    env: Mapping[str, str] | None = None,
    whitelist: Iterable[str] | None = None,
) -> dict[str, str]:
    """Return a sanitized copy of *env* preserving only whitelisted variables."""

    source = env or os.environ
    allowed = set(_DEFAULT_WHITELIST)
    if whitelist:
        allowed.update(whitelist)
    return {k: v for k, v in source.items() if k in allowed}


def sanitize_worker_env_value(value: object | None) -> str:
    """Return a numeric string for worker env overrides or ``""`` when invalid."""

    if value is None:
        return ""
    try:
        raw = str(value).strip()
    except Exception:
        return ""
    if not raw:
        return ""
    return raw if raw.isdigit() else ""


def get_worker_env_override(
    key: str,
    *,
    fallback_keys: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    """Return a sanitized integer override for executor worker counts."""

    env_map = env or os.environ
    search_order = (key,)
    if fallback_keys:
        search_order += tuple(fallback_keys)

    for env_key in search_order:
        raw = env_map.get(env_key)
        sanitized = sanitize_worker_env_value(raw)
        if sanitized:
            try:
                return int(sanitized)
            except (TypeError, ValueError):
                continue
    return 0
