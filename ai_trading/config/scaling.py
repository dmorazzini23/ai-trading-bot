"""Helpers for building lightweight scaling config from environment."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from ai_trading.config.management import merged_env_snapshot

# Default maximum factor applied to ATR-based scaling.  Deployments may
# override via the ``AI_TRADING_TAKE_PROFIT_FACTOR`` environment variable.
DEFAULT_MAX_FACTOR: float = 2.0

_LEGACY_KEY_MAP: dict[str, str] = {
    "CAPITAL_CAP": "AI_TRADING_CAPITAL_CAP",
    "TAKE_PROFIT_FACTOR": "AI_TRADING_TAKE_PROFIT_FACTOR",
}


@dataclass
class ScalingConfig:
    """Minimal subset of trading config used for sizing helpers."""

    capital_cap: float = 0.25
    max_factor: float = DEFAULT_MAX_FACTOR
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
    source_env = env or merged_env_snapshot()
    env_map = {k.upper(): v for k, v in source_env.items()}

    legacy_used = [
        f"{legacy}->{canonical}"
        for legacy, canonical in _LEGACY_KEY_MAP.items()
        if env_map.get(legacy) not in (None, "")
    ]
    if legacy_used:
        raise ValueError(
            "Deprecated scaling env keys are not supported. "
            f"Use canonical keys only: {', '.join(sorted(legacy_used))}"
        )

    capital_cap = float(env_map.get("AI_TRADING_CAPITAL_CAP", "0.25"))
    max_factor = float(
        env_map.get("AI_TRADING_TAKE_PROFIT_FACTOR", str(DEFAULT_MAX_FACTOR))
    )

    extras: dict[str, Any] = {}
    extras_raw = env_map.get("TRADING_CONFIG_EXTRAS")
    if extras_raw:
        try:
            parsed = json.loads(extras_raw)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError("TRADING_CONFIG_EXTRAS must be valid JSON") from exc
        if isinstance(parsed, dict):
            extras.update(parsed)

    known = {
        "AI_TRADING_CAPITAL_CAP",
        "AI_TRADING_TAKE_PROFIT_FACTOR",
        "TRADING_CONFIG_EXTRAS",
    }
    for k, v in env_map.items():
        if k not in known:
            extras[k] = _coerce(v)

    return ScalingConfig(capital_cap=capital_cap, max_factor=max_factor, extras=extras)


__all__ = ["ScalingConfig", "from_env", "DEFAULT_MAX_FACTOR"]
