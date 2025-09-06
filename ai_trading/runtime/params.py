"""Helpers for constructing runtime parameter dictionaries."""

from __future__ import annotations

from typing import Any, Mapping

from ai_trading.core.runtime import REQUIRED_PARAM_DEFAULTS


def build_runtime(overrides: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Merge runtime parameter defaults with overrides.

    Parameters with ``None`` values in ``overrides`` are ignored, leaving the
    existing/default value intact.

    Args:
        overrides: Optional mapping of parameter overrides.

    Returns:
        dict[str, float]: Merged runtime parameters where all values are floats.
    """
    params: dict[str, float] = {k: float(v) for k, v in REQUIRED_PARAM_DEFAULTS.items()}

    if overrides:
        for key, value in overrides.items():
            if value is None:
                continue
            params[key] = float(value)

    return params
