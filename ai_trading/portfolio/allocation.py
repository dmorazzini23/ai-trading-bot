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


def _normalize_bounded(
    weights: Mapping[str, float],
    *,
    min_weight: float,
    max_weight: float,
) -> dict[str, float]:
    if not weights:
        return {}
    low = max(0.0, float(min_weight))
    high = max(low, float(max_weight))
    names = list(weights)
    count = len(names)
    if count <= 0:
        return {}
    if (low * count) > 1.0 or (high * count) < 1.0:
        equal = 1.0 / count
        return {name: equal for name in names}

    result = {name: low for name in names}
    headroom = {name: high - low for name in names}
    remaining = 1.0 - (low * count)
    demand = {
        name: max(0.0, _clamp(float(weights[name]), low, high) - low)
        for name in names
    }
    active = {name for name in names if headroom[name] > 0.0}
    while remaining > 1e-12 and active:
        total_demand = sum(demand[name] for name in active)
        equal_share = total_demand <= 0.0
        total_basis = float(len(active)) if equal_share else total_demand
        progressed = 0.0
        saturated: set[str] = set()
        for name in list(active):
            basis = 1.0 if equal_share else demand[name]
            add = min(headroom[name], remaining * (basis / total_basis))
            if add <= 0.0:
                if headroom[name] <= 1e-12:
                    saturated.add(name)
                continue
            result[name] += add
            headroom[name] -= add
            progressed += add
            if headroom[name] <= 1e-12:
                saturated.add(name)
        remaining -= progressed
        active.difference_update(saturated)
        if progressed <= 1e-12:
            break

    total = sum(result.values())
    if total <= 0.0:
        equal = 1.0 / count
        return {name: equal for name in names}
    return {name: value / total for name, value in result.items()}


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

    return _normalize_bounded(updated, min_weight=min_weight, max_weight=max_weight)


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
