"""Bounded execution cost model calibrated from realized TCA records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    quantile = max(0.0, min(1.0, q))
    if len(ordered) == 1:
        return float(ordered[0])
    raw_index = quantile * (len(ordered) - 1)
    low = int(math.floor(raw_index))
    high = int(math.ceil(raw_index))
    if low == high:
        return float(ordered[low])
    ratio = raw_index - low
    return float((1.0 - ratio) * ordered[low] + ratio * ordered[high])


@dataclass(frozen=True, slots=True)
class CostModelParameters:
    """Serializable cost model parameters."""

    version: str
    base_cost_bps: float = 4.0
    spread_weight: float = 0.35
    volatility_weight: float = 0.75
    participation_weight: float = 1.25
    tca_weight: float = 0.45
    min_bps: float = 2.0
    max_bps: float = 25.0
    sample_count: int = 0
    calibrated_at: str | None = None


class CostModel:
    """Execution cost estimator with bounded output."""

    def __init__(self, params: CostModelParameters | None = None) -> None:
        self.params = params or CostModelParameters(version="v1")

    def estimate_cost_bps(
        self,
        *,
        spread_bps: float | None,
        volatility_pct: float | None,
        participation_rate: float | None,
        tca_cost_bps: float | None = None,
    ) -> float:
        spread = float(spread_bps) if spread_bps is not None else 0.0
        if not math.isfinite(spread):
            spread = 0.0
        spread = max(0.0, spread)

        vol = float(volatility_pct) if volatility_pct is not None else 0.0
        if not math.isfinite(vol):
            vol = 0.0
        vol = max(0.0, vol)

        participation = (
            float(participation_rate) if participation_rate is not None else 0.0
        )
        if not math.isfinite(participation):
            participation = 0.0
        participation = max(0.0, participation)

        model_estimate = (
            self.params.base_cost_bps
            + self.params.spread_weight * spread
            + self.params.volatility_weight * (vol * 100.0)
            + self.params.participation_weight * math.sqrt(participation * 100.0)
        )
        if tca_cost_bps is not None and math.isfinite(float(tca_cost_bps)):
            blend_w = max(0.0, min(1.0, self.params.tca_weight))
            model_estimate = ((1.0 - blend_w) * model_estimate) + (
                blend_w * float(tca_cost_bps)
            )

        bounded = max(
            self.params.min_bps,
            min(self.params.max_bps, float(model_estimate)),
        )
        return float(bounded)

    def calibrate(
        self,
        records: list[Mapping[str, Any]],
        *,
        min_samples: int = 30,
        quantile: float = 0.55,
        outlier_bps: float = 120.0,
    ) -> CostModelParameters:
        """Calibrate model parameters from realized TCA rows."""

        costs: list[float] = []
        spreads: list[float] = []
        vols: list[float] = []
        for record in records:
            raw_cost = record.get("is_bps")
            if raw_cost is None:
                continue
            try:
                cost = abs(float(raw_cost))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(cost) or cost <= 0:
                continue
            if cost > float(outlier_bps):
                continue
            costs.append(cost)
            try:
                spread = float(record.get("spread_paid_bps", 0.0) or 0.0)
            except (TypeError, ValueError):
                spread = 0.0
            if math.isfinite(spread) and spread > 0:
                spreads.append(spread)
            try:
                vol = float(record.get("volatility_pct", 0.0) or 0.0)
            except (TypeError, ValueError):
                vol = 0.0
            if math.isfinite(vol) and vol > 0:
                vols.append(vol)

        if len(costs) < max(1, int(min_samples)):
            return self.params

        median_cost = median(costs)
        q_cost = _percentile(costs, quantile)
        median_spread = median(spreads) if spreads else 0.0
        median_vol = median(vols) if vols else 0.0

        span = max(self.params.max_bps - self.params.min_bps, 1.0)
        normalized_cost = max(
            self.params.min_bps,
            min(self.params.max_bps, q_cost),
        )
        spread_weight = max(
            0.1,
            min(2.5, (median_spread / max(normalized_cost, 1.0))),
        )
        volatility_weight = max(
            0.25,
            min(2.0, (median_vol * 100.0) / max(span, 1.0)),
        )
        participation_weight = max(
            0.25,
            min(2.0, 0.75 + (median_cost / max(span, 1.0))),
        )

        updated = CostModelParameters(
            version=f"calibrated-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
            base_cost_bps=float(normalized_cost),
            spread_weight=float(spread_weight),
            volatility_weight=float(volatility_weight),
            participation_weight=float(participation_weight),
            tca_weight=self.params.tca_weight,
            min_bps=self.params.min_bps,
            max_bps=self.params.max_bps,
            sample_count=len(costs),
            calibrated_at=datetime.now(UTC).isoformat(),
        )
        self.params = updated
        return updated

    def to_dict(self) -> dict[str, Any]:
        """Return serializable representation."""

        return asdict(self.params)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CostModel":
        """Construct model from serialized payload."""

        params = CostModelParameters(
            version=str(payload.get("version", "v1")),
            base_cost_bps=float(payload.get("base_cost_bps", 4.0)),
            spread_weight=float(payload.get("spread_weight", 0.35)),
            volatility_weight=float(payload.get("volatility_weight", 0.75)),
            participation_weight=float(payload.get("participation_weight", 1.25)),
            tca_weight=float(payload.get("tca_weight", 0.45)),
            min_bps=float(payload.get("min_bps", 2.0)),
            max_bps=float(payload.get("max_bps", 25.0)),
            sample_count=int(payload.get("sample_count", 0)),
            calibrated_at=(
                str(payload.get("calibrated_at"))
                if payload.get("calibrated_at") is not None
                else None
            ),
        )
        return cls(params=params)

    @classmethod
    def load(cls, path: str | Path) -> "CostModel":
        """Load model from JSON file, returning defaults on failure."""

        target = Path(path)
        if not target.exists():
            return cls()
        try:
            with target.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return cls()
        if not isinstance(payload, Mapping):
            return cls()
        return cls.from_dict(payload)

    def save(self, path: str | Path) -> None:
        """Persist model parameters as JSON."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, sort_keys=True, indent=2)

