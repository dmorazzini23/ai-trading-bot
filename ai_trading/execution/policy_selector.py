"""Execution policy selector based on urgency, spread, and liquidity context."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any


class ExecutionPolicy(str, Enum):
    """Supported high-level execution policies."""

    PASSIVE_LIMIT = "passive_limit"
    MARKETABLE_LIMIT = "marketable_limit"
    TWAP = "twap"
    POV = "pov"


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Selector result with rationale."""

    policy: ExecutionPolicy
    reasons: tuple[str, ...]
    participation_rate: float
    urgency_score: float


class ExecutionPolicySelector:
    """Rule-based selector for execution policy routing."""

    def __init__(
        self,
        *,
        spread_wide_bps: float = 18.0,
        spread_tight_bps: float = 6.0,
        high_urgency: float = 0.75,
        twap_participation: float = 0.08,
        pov_participation: float = 0.03,
    ) -> None:
        self.spread_wide_bps = float(spread_wide_bps)
        self.spread_tight_bps = float(spread_tight_bps)
        self.high_urgency = float(high_urgency)
        self.twap_participation = float(twap_participation)
        self.pov_participation = float(pov_participation)

    @staticmethod
    def _normalize_urgency(urgency: str | float | int | None) -> float:
        if urgency is None:
            return 0.5
        if isinstance(urgency, str):
            token = urgency.strip().lower()
            mapping = {
                "low": 0.25,
                "normal": 0.5,
                "medium": 0.5,
                "high": 0.8,
                "urgent": 0.95,
            }
            if token in mapping:
                return mapping[token]
            try:
                raw = float(token)
            except ValueError:
                return 0.5
            return max(0.0, min(1.0, raw))
        try:
            raw = float(urgency)
        except (TypeError, ValueError):
            return 0.5
        if not math.isfinite(raw):
            return 0.5
        return max(0.0, min(1.0, raw))

    def select_policy(
        self,
        *,
        spread_bps: float | None,
        volatility_pct: float | None,
        order_notional: float | None,
        avg_daily_volume_notional: float | None,
        urgency: str | float | int | None,
        data_provenance: str | None = None,
        allow_twap: bool = True,
    ) -> PolicyDecision:
        """Select the execution policy with deterministic rationale."""

        spread = float(spread_bps) if spread_bps is not None else 10.0
        if not math.isfinite(spread):
            spread = 10.0
        spread = max(0.0, spread)

        vol = float(volatility_pct) if volatility_pct is not None else 0.0
        if not math.isfinite(vol):
            vol = 0.0
        vol = max(0.0, vol)

        notional = float(order_notional) if order_notional is not None else 0.0
        if not math.isfinite(notional):
            notional = 0.0
        notional = max(0.0, notional)

        adv_notional = (
            float(avg_daily_volume_notional)
            if avg_daily_volume_notional is not None
            else 0.0
        )
        if not math.isfinite(adv_notional):
            adv_notional = 0.0
        adv_notional = max(0.0, adv_notional)
        participation = notional / max(adv_notional, 1.0)

        urgency_score = self._normalize_urgency(urgency)
        provenance = str(data_provenance or "").strip().lower()
        delayed_feed = provenance in {"delayed_sip", "delayed", "unknown"}

        reasons: list[str] = []

        if allow_twap and participation >= self.twap_participation:
            reasons.append("participation_above_twap_threshold")
            return PolicyDecision(
                policy=ExecutionPolicy.TWAP,
                reasons=tuple(reasons),
                participation_rate=participation,
                urgency_score=urgency_score,
            )

        if participation >= self.pov_participation:
            reasons.append("participation_above_pov_threshold")
            return PolicyDecision(
                policy=ExecutionPolicy.POV,
                reasons=tuple(reasons),
                participation_rate=participation,
                urgency_score=urgency_score,
            )

        if delayed_feed and urgency_score < self.high_urgency:
            reasons.append("delayed_provenance_prefers_passive")
            return PolicyDecision(
                policy=ExecutionPolicy.PASSIVE_LIMIT,
                reasons=tuple(reasons),
                participation_rate=participation,
                urgency_score=urgency_score,
            )

        if spread >= self.spread_wide_bps and urgency_score < self.high_urgency:
            reasons.append("wide_spread_passive_preferred")
            return PolicyDecision(
                policy=ExecutionPolicy.PASSIVE_LIMIT,
                reasons=tuple(reasons),
                participation_rate=participation,
                urgency_score=urgency_score,
            )

        if urgency_score >= self.high_urgency or spread <= self.spread_tight_bps:
            reasons.append("urgent_or_tight_spread_marketable")
            return PolicyDecision(
                policy=ExecutionPolicy.MARKETABLE_LIMIT,
                reasons=tuple(reasons),
                participation_rate=participation,
                urgency_score=urgency_score,
            )

        if vol >= 0.03:
            reasons.append("high_volatility_reduce_crossing_risk")
            policy = ExecutionPolicy.PASSIVE_LIMIT
        else:
            reasons.append("balanced_default_pov")
            policy = ExecutionPolicy.POV

        return PolicyDecision(
            policy=policy,
            reasons=tuple(reasons),
            participation_rate=participation,
            urgency_score=urgency_score,
        )


def select_execution_policy(**kwargs: Any) -> PolicyDecision:
    """Convenience helper using default selector parameters."""

    return ExecutionPolicySelector().select_policy(**kwargs)

