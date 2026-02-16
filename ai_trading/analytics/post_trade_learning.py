"""Bounded post-trade learning updates from outcomes and TCA."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping


@dataclass(slots=True)
class LearningBounds:
    max_daily_delta_bps: float
    max_daily_delta_frac: float


def _bounded_delta(value: float, limit: float) -> float:
    return max(-abs(limit), min(abs(limit), value))


def compute_learning_updates(
    *,
    symbol_metrics: Mapping[str, Mapping[str, float]],
    bounds: LearningBounds,
    is_bps_trigger: float,
    flip_rate_trigger: float,
) -> dict[str, Any]:
    per_symbol_cost_buffer_bps: dict[str, float] = {}
    global_deadband_frac_delta = 0.0
    for symbol, metrics in symbol_metrics.items():
        is_bps = float(metrics.get("is_bps", 0.0))
        flip_rate = float(metrics.get("flip_rate", 0.0))
        if is_bps > is_bps_trigger:
            delta_bps = _bounded_delta(is_bps - is_bps_trigger, bounds.max_daily_delta_bps)
            per_symbol_cost_buffer_bps[str(symbol)] = float(delta_bps)
        if flip_rate > flip_rate_trigger:
            global_deadband_frac_delta = max(
                global_deadband_frac_delta,
                _bounded_delta(flip_rate - flip_rate_trigger, bounds.max_daily_delta_frac),
            )

    updates = {
        "timestamp": datetime.now(UTC).isoformat(),
        "overrides": {
            "per_symbol_cost_buffer_bps": per_symbol_cost_buffer_bps,
            "global_deadband_frac_delta": global_deadband_frac_delta,
        },
        "rollback": {
            "created_ts": datetime.now(UTC).isoformat(),
            "expires_ts": (datetime.now(UTC) + timedelta(days=30)).isoformat(),
        },
    }
    return updates


def write_learning_overrides(path: str, updates: Mapping[str, Any]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(dict(updates), sort_keys=True), encoding="utf-8")


def load_learning_overrides(path: str, *, max_age_days: int = 30) -> dict[str, Any]:
    src = Path(path)
    if not src.exists():
        return {}
    try:
        payload = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    ts_raw = payload.get("timestamp")
    if not ts_raw:
        return {}
    try:
        ts = datetime.fromisoformat(str(ts_raw))
    except ValueError:
        return {}
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    if datetime.now(UTC) - ts > timedelta(days=max(1, int(max_age_days))):
        return {}
    return payload.get("overrides", {}) if isinstance(payload.get("overrides"), dict) else {}
