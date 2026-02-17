from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


class PresetValidationError(ValueError):
    """Raised when operator preset input violates guardrails."""


@dataclass(frozen=True, slots=True)
class StrategyPreset:
    """No-code strategy preset with bounded risk defaults."""

    name: str
    description: str
    trading_mode: str
    confidence_threshold: float
    capital_cap: float
    dollar_risk_limit: float
    max_positions: int
    max_position_size: float


_PRESETS: tuple[StrategyPreset, ...] = (
    StrategyPreset(
        name="conservative",
        description="Lower turnover and tighter loss controls.",
        trading_mode="conservative",
        confidence_threshold=0.80,
        capital_cap=0.03,
        dollar_risk_limit=0.02,
        max_positions=6,
        max_position_size=10000.0,
    ),
    StrategyPreset(
        name="balanced",
        description="Default balanced profile for steady deployment.",
        trading_mode="balanced",
        confidence_threshold=0.75,
        capital_cap=0.06,
        dollar_risk_limit=0.05,
        max_positions=10,
        max_position_size=25000.0,
    ),
    StrategyPreset(
        name="aggressive",
        description="Higher throughput with larger allocation limits.",
        trading_mode="aggressive",
        confidence_threshold=0.70,
        capital_cap=0.10,
        dollar_risk_limit=0.08,
        max_positions=14,
        max_position_size=40000.0,
    ),
)

_RANGE_FLOAT: dict[str, tuple[float, float]] = {
    "confidence_threshold": (0.50, 0.95),
    "capital_cap": (0.01, 0.25),
    "dollar_risk_limit": (0.01, 0.20),
    "max_position_size": (1000.0, 1000000.0),
}

_RANGE_INT: dict[str, tuple[int, int]] = {
    "max_positions": (1, 50),
}


def list_presets() -> list[dict[str, Any]]:
    """Return operator presets as JSON-serializable dictionaries."""

    return [asdict(item) for item in _PRESETS]


def _coerce_float(key: str, value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PresetValidationError(f"{key} must be numeric") from exc
    lower, upper = _RANGE_FLOAT[key]
    if parsed < lower or parsed > upper:
        raise PresetValidationError(f"{key} must be between {lower} and {upper}")
    return parsed


def _coerce_int(key: str, value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise PresetValidationError(f"{key} must be an integer") from exc
    lower, upper = _RANGE_INT[key]
    if parsed < lower or parsed > upper:
        raise PresetValidationError(f"{key} must be between {lower} and {upper}")
    return parsed


def build_plan(
    preset_name: str,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build execution-ready config from a preset and optional guarded overrides."""

    normalized = str(preset_name or "").strip().lower()
    selected = next((item for item in _PRESETS if item.name == normalized), None)
    if selected is None:
        supported = ", ".join(item.name for item in _PRESETS)
        raise PresetValidationError(f"Unknown preset '{preset_name}'. Supported: {supported}")

    plan = asdict(selected)
    for key, value in dict(overrides or {}).items():
        if key in _RANGE_FLOAT:
            plan[key] = _coerce_float(key, value)
        elif key in _RANGE_INT:
            plan[key] = _coerce_int(key, value)

    return plan


__all__ = ["PresetValidationError", "StrategyPreset", "build_plan", "list_presets"]
