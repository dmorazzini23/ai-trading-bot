"""Execution helpers for isolated subprocess and worker environments."""
from __future__ import annotations

import os
from typing import Iterable, Mapping

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
    """Return a sanitized copy of *env* preserving only whitelisted variables.

    Parameters
    ----------
    env:
        Optional source environment mapping. Defaults to :data:`os.environ`.
    whitelist:
        Additional variable names to preserve.
    """
    source = env or os.environ
    allowed = set(_DEFAULT_WHITELIST)
    if whitelist:
        allowed.update(whitelist)
    return {k: v for k, v in source.items() if k in allowed}
