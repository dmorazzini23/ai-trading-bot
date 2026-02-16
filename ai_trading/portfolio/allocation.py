"""Sleeve-level capital allocation state and bounded updates."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


@dataclass(slots=True)
class SleevePerfState:
    rolling_expectancy: float
    drawdown: float
    stability_score: float
    trade_count: int
    confidence: float


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def update_allocation_weights(
    *,
    base_weights: Mapping[str, float],
    perf_states: Mapping[str, SleevePerfState],
    min_weight: float,
    max_weight: float,
    daily_max_delta: float,
    expectancy_floor: float,
    drawdown_trigger: float,
    min_trades_for_adjust: int,
) -> dict[str, float]:
    updated: dict[str, float] = {name: float(weight) for name, weight in base_weights.items()}
    for sleeve, base in list(updated.items()):
        perf = perf_states.get(sleeve)
        if perf is None or perf.trade_count < int(min_trades_for_adjust):
            updated[sleeve] = _clamp(base, min_weight, max_weight)
            continue
        delta = 0.0
        if perf.rolling_expectancy < expectancy_floor or perf.drawdown > drawdown_trigger:
            delta = -abs(daily_max_delta)
        elif perf.rolling_expectancy > 0 and perf.stability_score > 0.5 and perf.confidence > 0.5:
            delta = abs(daily_max_delta)
        updated[sleeve] = _clamp(base + delta, min_weight, max_weight)

    total = sum(updated.values())
    if total <= 0:
        equal = 1.0 / max(1, len(updated))
        return {sleeve: equal for sleeve in updated}
    return {sleeve: weight / total for sleeve, weight in updated.items()}


def save_allocation_state(path: str, weights: Mapping[str, float], *, metadata: Mapping[str, Any] | None = None) -> None:
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "weights": dict(weights),
        "metadata": dict(metadata or {}),
    }
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def load_allocation_state(path: str) -> dict[str, float]:
    src = Path(path)
    if not src.exists():
        return {}
    try:
        payload = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    weights = payload.get("weights")
    if not isinstance(weights, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in weights.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out
