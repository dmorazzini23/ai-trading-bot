"""Helpers for building lightweight scaling config from environment."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class ScalingConfig:
    """Minimal subset of trading config used for sizing helpers."""

    capital_cap: float = 0.04
    extras: dict[str, Any] | None = None

    def derive_cap_from_settings(self, equity: float, fallback: float = 8000.0) -> float:
        """Return max position size based on equity and ``capital_cap``."""
        if equity and equity > 0:
            return float(equity) * float(self.capital_cap)
        return float(fallback)


def _coerce(value: str) -> Any:
    """Coerce a string to int or float when possible."""
    try:
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except Exception:
        return value


def from_env(env: Mapping[str, str] | None = None) -> ScalingConfig:
    """Build :class:`ScalingConfig` from environment mapping.

    Unknown keys are collected into ``extras``.  When ``extras`` is empty
    it is initialised to an empty dict rather than ``None``.  Numeric values
    within ``extras`` are coerced to ``int`` or ``float`` for convenience.
    """
    env_map = {k.upper(): v for k, v in (env or os.environ).items()}

    capital_cap = float(env_map.get("CAPITAL_CAP", "0.04"))

    extras: dict[str, Any] = {}
    extras_raw = env_map.get("TRADING_CONFIG_EXTRAS")
    if extras_raw:
        try:
            parsed = json.loads(extras_raw)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError("TRADING_CONFIG_EXTRAS must be valid JSON") from exc
        if isinstance(parsed, dict):
            extras.update(parsed)

    known = {"CAPITAL_CAP", "TRADING_CONFIG_EXTRAS"}
    for k, v in env_map.items():
        if k not in known:
            extras[k] = _coerce(v)

    return ScalingConfig(capital_cap=capital_cap, extras=extras)


__all__ = ["ScalingConfig", "from_env"]
